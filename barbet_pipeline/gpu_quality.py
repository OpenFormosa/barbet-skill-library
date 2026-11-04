"""Frozen Barbet likelihood and detached-readout gradient measurements."""
from __future__ import annotations
import math
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def readout_sketch(probabilities, targets, hidden, vocabulary_projection, hidden_projection):
    residual = probabilities @ vocabulary_projection - vocabulary_projection[targets]
    projected_hidden = hidden @ hidden_projection
    return residual.T @ projected_hidden


class FrozenScorer:
    def __init__(self, model, *, projection_seed=20261104, sketch=True):
        self.model=model;self.device=model.model.embed_tokens.weight.device
        self.sketch=sketch
        if sketch:
            generator=torch.Generator().manual_seed(projection_seed)
            vocab,dim=model.lm_head.weight.shape
            self.vocabulary_projection=(torch.randint(0,2,(vocab,64),generator=generator,dtype=torch.int8).float()*2-1).to(self.device)/math.sqrt(64)
            self.hidden_projection=(torch.randint(0,2,(dim,128),generator=generator,dtype=torch.int8).float()*2-1).to(self.device)/math.sqrt(128)

    @torch.inference_mode()
    def score(self, rows, *, projection=True, projection_chunk=256):
        if not rows:raise ValueError('Empty score batch')
        length=max(len(row['input_ids']) for row in rows)
        if length>self.model.config.max_position_embeddings:raise ValueError('No prompt truncation permitted')
        inputs=torch.full((len(rows),length),self.model.config.eos_token_id,dtype=torch.long,device=self.device)
        for index,row in enumerate(rows):
            if len(row['input_ids'])!=len(row['labels']) or row['labels'][0]!=-100:raise ValueError('Unshifted assistant mask required')
            inputs[index,:len(row['input_ids'])]=torch.tensor(row['input_ids'],device=self.device)
        # Right padding is strictly after each real sequence. Causal attention and
        # causal Mamba cannot expose padding to any supervised real position.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            hidden=self.model.model(input_ids=inputs,use_cache=False,return_dict=True).last_hidden_state
        output=[]
        for index,row in enumerate(rows):
            labels=torch.tensor(row['labels'],device=self.device)
            positions=torch.nonzero(labels!=-100,as_tuple=True)[0]
            if not len(positions):raise ValueError('No supervised targets')
            target=labels[positions];vectors=hidden[index,positions-1]
            loss_sum=torch.zeros((),device=self.device);correct=torch.zeros((),dtype=torch.long,device=self.device)
            feature=torch.zeros((64,128),device=self.device) if projection and self.sketch else None
            for begin in range(0,len(positions),projection_chunk):
                y=target[begin:begin+projection_chunk];h=vectors[begin:begin+projection_chunk]
                logits=self.model.lm_head(h).float()
                logp=torch.log_softmax(logits,dim=-1)
                loss_sum-=logp.gather(1,y[:,None]).sum();correct+=(logits.argmax(-1)==y).sum()
                if feature is not None:
                    feature+=readout_sketch(logp.exp(),y,h.float(),self.vocabulary_projection,self.hidden_projection)
            if not torch.isfinite(loss_sum):raise ValueError('Nonfinite target NLL')
            result={'record_id':row['record_id'],'input_tokens':len(row['input_ids']),
                    'supervised_tokens':len(positions),'nll_sum':float(loss_sum),'nll':float(loss_sum/len(positions)),
                    'teacher_forced_top1_correct':int(correct)}
            if feature is not None:
                feature/=len(positions);norm=feature.norm()
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient sketch')
                result['readout_gradient_sketch_norm']=float(norm)
                result['_feature']=(feature.flatten()/norm.clamp_min(1e-12)).cpu()
            output.append(result)
        return output
