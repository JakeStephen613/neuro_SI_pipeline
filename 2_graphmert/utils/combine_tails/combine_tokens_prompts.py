MEANING_EXPL_NEURO = """Relation meanings (neuroscience KG). Use the direction exactly as written.

Core structure / hierarchy
* part_of - component → whole: “s4 segment” part_of “voltage-gated sodium channel”
* contains - whole → component: “hippocampus” contains “dentate gyrus”
* located_in - entity → location: “pyramidal neuron” located_in “cortical layer v”
* connected_to - undirected anatomical or functional connection: “primary motor cortex” connected_to “premotor cortex”

Circuit flow / anatomy of projections
* projects_to - source sends axons to target: “retina” projects_to “lateral geniculate nucleus”
* receives_input_from - target receives driving input from source: “lateral geniculate nucleus” receives_input_from “retinal ganglion cells”
* innervates - neuron or population supplies synapses to structure: “motor neuron” innervates “skeletal muscle”
* originates_from - pathway or tract begins in: “corticospinal tract” originates_from “primary motor cortex”

Molecular / cellular interactions
* releases - cell or terminal releases molecule: “dopaminergic neuron” releases “dopamine”
* binds_to - ligand binds receptor or molecular target: “glutamate” binds_to “NMDA receptor”
* activates - source increases activity or open probability of target: “glutamate” activates “AMPA receptor”
* inhibits - source decreases activity or firing of target: “GABA” inhibits “pyramidal neuron”
* modulates - source alters gain, kinetics, or responsiveness: “acetylcholine” modulates “cortical oscillations”
* regulates - broader control of expression, function, or excitability: “SCN1A mutation” regulates “Nav1.1 channel function”
* transports - transporter or channel moves substance: “voltage-gated sodium channel” transports “sodium ions”
* forms_complex_with - physical molecular complex formation: “SNARE proteins” forms_complex_with “synaptotagmin”

Dynamics / representation
* responds_to - entity responds to stimulus or condition: “auditory cortex” responds_to “sound”
* encodes_representation_of - neural activity represents a variable: “hippocampal place cells” encodes_representation_of “spatial location”
* represents - entity stands for or reflects information: “firing rate” represents “stimulus intensity”

Pathways / dependency
* participates_in - entity takes part in a process or pathway: “voltage-gated sodium channels” participates_in “action potential”
* mediates_signal_for - entity mediates a physiological signal component: “voltage-gated sodium channels” mediates_signal_for “rising phase of action potential”
* required_for - necessary condition for a process: “voltage-gated sodium channels” required_for “action potential initiation”
* causes - direct causal relationship: “SCN1A loss-of-function mutation” causes “Dravet syndrome”
* results_in - downstream outcome or effect: “inward sodium current” results_in “membrane depolarization”

Clinical linkage
* associated_with - correlated without direct causality: “hippocampal atrophy” associated_with “Alzheimer’s disease”
* symptom_of - symptom belongs to disorder: “seizures” symptom_of “Dravet syndrome”
* impaired_in - function reduced in condition: “working memory” impaired_in “schizophrenia”
* expressed_in - gene or protein expressed in tissue or cell type: “Nav1.1” expressed_in “GABAergic interneurons”
* controls - entity exerts control over process or variable: “basal ganglia output” controls “movement initiation”
"""

SYSTEM_CONTEXT_TEMPLATE = """You are an expert Neuroscience Knowledge Graph curator. You specialize in identifying highly specific biological entities.

You will be given:
- Context text (may be empty)
- A head entity
- A relation
- A list of candidate tokens ("predictions")

### YOUR PRIMARY DIRECTIVE:
Maximize SPECIFICITY. Do not use a generic term (e.g., "neuron", "cortex", "receptor") if the candidates allow you to build a specific term (e.g., "parvalbumin-positive GABAergic interneuron").

### RULES:
1) Output ONLY a JSON list of strings.
2) COMBINE tokens to reconstruct full technical names. If candidates contain "NMDA", "receptor", and "subunit", you SHOULD output ["NMDA receptor subunit"].
3) DISCARD vague or redundant candidates that add no specific value.
4) Respect the biological directionality of the relation.
5) If multiple specific entities are present, include them all.

Relation meanings:
{MEANING_EXPL}
"""

SYSTEM_CONTEXT = SYSTEM_CONTEXT_TEMPLATE.format(MEANING_EXPL=MEANING_EXPL_NEURO)
# Example 1: Combining physiological components
example_user_1 = """Input:
voltage-gated sodium channels are responsible for the rising phase of the action potential.
head: voltage-gated sodium channels
relation: mediates_signal_for
predictions: ["rising phase", "action potential", "rising phase of action potential", "signal"]
Output:"""
example_assistant_1 = """["rising phase of action potential"]"""

# Example 2: Structural precision
example_user_2 = """Input:
the s4 segment acts as the primary voltage sensor within the sodium channel pore.
head: s4 segment
relation: part_of
predictions: ["voltage-gated sodium channel", "channel", "pore", "voltage-gated sodium channel pore"]
Output:"""
example_assistant_2 = """["voltage-gated sodium channel pore"]"""

# Example 3: Precise biophysical outcomes
example_user_3 = """Input:
the rapid influx of sodium ions leads to a localized depolarization of the axonal membrane.
head: inward sodium current
relation: results_in
predictions: ["depolarization", "membrane", "axonal membrane depolarization", "axonal"]
Output:"""
example_assistant_3 = """["axonal membrane depolarization"]"""

# Example 4: Identifying specific cell types
example_user_4 = """Input:
gaba released from fast-spiking interneurons inhibits layer v pyramidal neurons.
head: gaba
relation: inhibits
predictions: ["neurons", "pyramidal neurons", "layer v pyramidal neurons", "cortex"]
Output:"""
example_assistant_4 = """["layer v pyramidal neurons"]"""

# Example 5: Molecular specificity
example_user_5 = """Input:
midbrain dopaminergic neurons release dopamine into the ventral striatum.
head: midbrain dopaminergic neurons
relation: releases
predictions: ["dopamine", "neurotransmitter", "chemical", "monoamine"]
Output:"""
example_assistant_5 = """["dopamine"]"""

# Example 6: Circuit source identification
example_user_6 = """Input:
the lateral geniculate nucleus receives driving input from m-type retinal ganglion cells.
head: lateral geniculate nucleus
relation: receives_input_from
predictions: ["retinal ganglion cells", "m-type retinal ganglion cells", "retina", "cells"]
Output:"""
example_assistant_6 = """["m-type retinal ganglion cells"]"""

# Example 7: Clinical phenotype precision
example_user_7 = """Input:
loss of scn1a function is the primary driver of severe myoclonic epilepsy in infancy, also known as dravet syndrome.
head: scn1a mutation
relation: causes
predictions: ["dravet syndrome", "severe myoclonic epilepsy in infancy", "epilepsy", "syndrome"]
Output:"""
example_assistant_7 = """["severe myoclonic epilepsy in infancy", "dravet syndrome"]"""

# Example 8: Representational precision
example_user_8 = """Input:
hippocampal place cells encode the animal's precise spatial coordinate within the environment.
head: hippocampal place cells
relation: encodes_representation_of
predictions: ["spatial location", "precise spatial coordinate", "location", "space"]
Output:"""
example_assistant_8 = """["precise spatial coordinate"]"""

# Example 9: Complex disorder naming
example_user_9 = """Input:
working memory impairment is a hallmark of patients diagnosed with paranoid schizophrenia.
head: working memory
relation: impaired_in
predictions: ["schizophrenia", "paranoid schizophrenia", "disorder", "psychosis"]
Output:"""
example_assistant_9 = """["paranoid schizophrenia"]"""