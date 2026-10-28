"""Strict right-padding for pretokenized Barbet assistant-only SFT examples."""
from dataclasses import dataclass


@dataclass
class BarbetSFTCollator:
    pad_token_id: int = 114691
    bos_token_id: int = 114689
    eos_token_id: int = 114690
    vocab_size: int = 114944
    max_length: int = 131072

    def __call__(self, features):
        import torch
        if not features:
            raise ValueError('Empty training batch')
        lengths = []
        for row in features:
            ids, labels = row['input_ids'], row['labels']
            if (not ids or len(ids) != len(labels) or len(ids) > self.max_length
                    or ids[0] != self.bos_token_id or labels[0] != -100):
                raise ValueError('Invalid unshifted Barbet sequence')
            if any(type(i) is not int or not 0 <= i < self.vocab_size or i == self.pad_token_id for i in ids):
                raise ValueError('Invalid vocabulary ID or PAD inside real content')
            if any(type(y) is not int or y != -100 and y != x for x, y in zip(ids, labels)):
                raise ValueError('Labels must preserve input IDs at supervised positions')
            targets = [y for y in labels if y != -100]
            if not targets or targets[-1] != self.eos_token_id:
                raise ValueError('Missing assistant EOS supervision')
            lengths.append(len(ids))
        width = max(lengths)
        input_ids = torch.full((len(features), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), width), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(features), width), dtype=torch.long)
        for i, (row, length) in enumerate(zip(features, lengths)):
            input_ids[i, :length] = torch.tensor(row['input_ids'], dtype=torch.long)
            labels[i, :length] = torch.tensor(row['labels'], dtype=torch.long)
            attention_mask[i, :length] = 1
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
