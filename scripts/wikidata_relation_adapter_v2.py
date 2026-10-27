"""Add portable tool-state metadata without rewriting any reviewed messages."""
from dataclasses import asdict
from scripts import wikidata_relation_adapter_v1 as original
from barbet_inference.relational_memory import RelationalMemory

SOURCE=original.SOURCE
URL=original.URL
SKILLS=original.SKILLS
INVENTORY=original.INVENTORY
PRIOR=original.PRIOR


class SourceGraph(original.SourceGraph):
    def make_row(self,anchor,skills):
        row=super().make_row(anchor,skills)
        refs=row['source_refs']
        facts=self.facts([line-1 for line in refs['original_training_source_line_numbers']])
        memory=RelationalMemory(facts)
        if memory.snapshot_sha256!=refs['snapshot_sha256']:
            raise ValueError('metadata restoration differs from actual task snapshot')
        row['memory']['registered_relation_snapshot']={
            'format':'barbet-finite-relations-v1','tool_name':'relation_lookup',
            'visibility':'registered_tool_state; observations become model-visible only after calls',
            'snapshot_sha256':memory.snapshot_sha256,'facts':[asdict(f) for f in memory.facts]}
        row['source_refs']['portable_tool_state_added_without_message_or_target_change']=True
        return row
