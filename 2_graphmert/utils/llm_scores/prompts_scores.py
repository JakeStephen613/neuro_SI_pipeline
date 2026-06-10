system_prompt_validity_score = """You will evaluate the quality of triples for a neuroscience knowledge graph involving brain regions, cell types, molecular entities, neural circuits, physiological processes, and synaptic mechanisms.

For each triple, you are given:
- A head entity, a relation, and a tail entity

Your task: Accept the triple or reject it based on:
- Logical alignment: the tail must logically align with the head and relation
- Knowledge value: the triple must add at least some sorot of biologically meaningful information relevant to neuroscience
- only assign no if you are completely certain the triple is clearly incorrect, nonesensical, or is very trivial and does not contribute to KG in any way, be hesistant to give no

IMPORTANT OUTPUT FORMAT:
1. First, think through your reasoning inside <think> tags
2. Then, output EXACTLY one of these on a new line:
   - [yes] if the triple is valid
   - [no] if the triple is invalid

rethink every no you give, only give no if you are 100% sure the triple is invalid

Example:
<think>
Sodium channels open during depolarization, allowing sodium influx. This is a core mechanism of action potentials. The relation "causes" is appropriate here.
</think>
[yes]

Now evaluate the following triple:"""