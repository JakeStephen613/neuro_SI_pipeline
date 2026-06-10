#!/usr/bin/env python3


# =========================================================
# LARGEST LIST OF RELATION EXAMPLES (NEURO)
# =========================================================

EXAMPLES_gemini_score45 = """
hippocampus | part_of | limbic system
microglia | participates_in | neuroinflammation
serotonin | modulates | mood regulation
cerebral cortex | contains | pyramidal neurons
substantia nigra pars compacta | located_in | midbrain
primary motor cortex | projects_to | spinal cord
dopamine signaling | mediates_signal_for | reward prediction
alpha-synuclein aggregation | causes | dopaminergic neuron loss
NMDA receptors | required_for | long-term potentiation
basal ganglia output | controls | voluntary movement
prefrontal cortex function | impaired_in | schizophrenia
NMDA receptor | binds_to | glutamate
resting tremor | symptom_of | Parkinson disease
tau pathology | associated_with | Alzheimer disease
anterior cingulate cortex | connected_to | prefrontal cortex
lateral geniculate nucleus | receives_input_from | retina
demyelination | results_in | reduced conduction velocity
NMDA receptor activation | activates | calcium influx
GABA_A receptor activation | inhibits | neuronal firing
vagus nerve | innervates | gut
hippocampus | encodes_representation_of | episodic memory
hippocampal pyramidal neurons | responds_to | glutamate
dopamine D1 receptor | expressed_in | medium spiny neurons
motor cortex activity | represents | movement direction
BDNF | regulates | synaptic plasticity
SERT (serotonin transporter) | transports | serotonin
SNARE complex | forms_complex_with | synaptotagmin
dopaminergic neurons | releases | dopamine
nigrostriatal pathway | originates_from | substantia nigra pars compacta

"""

ALLOWED_RELATIONS_qwen32_score45 = """
    "part_of",
    "participates_in",
    "modulates",
    "contains",
    "located_in",
    "projects_to",
    "mediates_signal_for",
    "causes",
    "required_for",
    "controls",
    "impaired_in",
    "binds_to",
    "symptom_of",
    "associated_with",
    "connected_to",
    "receives_input_from",
    "results_in",
    "activates",
    "inhibits",
    "innervates",
    "encodes_representation_of",
    "responds_to",
    "expressed_in",
    "represents",
    "regulates",
    "transports",
    "originates_from",
    "forms_complex_with",
    "releases",
"""


EXAMPLES_qwen32_score45 = EXAMPLES_gemini_score45


ALLOWED_RELATIONS_qwen32_score55_exp_kg = ALLOWED_RELATIONS_qwen32_score45
EXAMPLES_qwen32_score55_exp_kg = EXAMPLES_gemini_score45


MEANING_EXPL_qwen32_score55_exp_kg = """These relations are neuroscience KG relations.

General structural / spatial relations:
* part_of - X is a component or subregion of Y.
  Example: hippocampus part_of limbic system.
* contains - Y contains X as a constituent structure or cell type.
  Example: cerebral cortex contains pyramidal neurons.
* located_in - X is anatomically located in Y.
  Example: substantia nigra pars compacta located_in midbrain.
* connected_to - anatomical or functional connectivity between regions.
  Example: anterior cingulate cortex connected_to prefrontal cortex.
* originates_from - pathway or structure begins at Y.
  Example: nigrostriatal pathway originates_from substantia nigra.

Circuit directionality:
* projects_to - axonal projections from X to Y.
  Example: primary motor cortex projects_to spinal cord.
* receives_input_from - X receives primary input from Y.
  Example: lateral geniculate nucleus receives_input_from retina.
* innervates - nerve or neuron X innervates target Y.
  Example: vagus nerve innervates gut.

Molecular and cellular relations:
* expressed_in - gene, receptor, or protein expressed in cell type or region.
  Example: dopamine D1 receptor expressed_in medium spiny neurons.
* releases - neuron or cell releases molecule.
  Example: dopaminergic neurons releases dopamine.
* binds_to - molecule binds to receptor or target.
  Example: NMDA receptor binds_to glutamate.
* forms_complex_with - proteins physically associate in a complex.
  Example: SNARE complex forms_complex_with synaptotagmin.
* transports - transporter moves molecule across membrane.
  Example: SERT transports serotonin.

Functional and regulatory relations:
* activates - X increases activity of Y.
  Example: NMDA receptor activation activates calcium influx.
* inhibits - X decreases activity of Y.
  Example: GABA_A receptor activation inhibits neuronal firing.
* modulates - X modulates Y in a context-dependent manner.
  Example: serotonin modulates mood regulation.
* regulates - X regulates Y over longer timescales.
  Example: BDNF regulates synaptic plasticity.
* responds_to - neuron or system responds to stimulus or ligand.
  Example: hippocampal pyramidal neurons responds_to glutamate.
* represents - neural activity represents a variable.
  Example: motor cortex activity represents movement direction.
* encodes_representation_of - region encodes representation of information.
  Example: hippocampus encodes_representation_of episodic memory.
* participates_in - entity participates in a biological process.
  Example: microglia participates_in neuroinflammation.
* required_for - X is necessary for Y.
  Example: NMDA receptors required_for long-term potentiation.
* mediates_signal_for - X mediates signaling for Y.
  Example: dopamine signaling mediates_signal_for reward prediction.
* controls - X controls Y (behavior, process, output).
  Example: basal ganglia output controls voluntary movement.
* regulates - X regulates Y (longer timescales).
  Example: BDNF regulates synaptic plasticity.
* transports - transporter moves molecule across membrane.
  Example: SERT transports serotonin.
* forms_complex_with - proteins physically associate in a complex.
  Example: SNARE complex forms_complex_with synaptotagmin.

Clinical and pathological relations:
* associated_with - non-directional association without causal commitment.
  Example: tau pathology associated_with Alzheimer disease.
* causes - X causes Y.
  Example: alpha-synuclein aggregation causes dopaminergic neuron loss.
* results_in - X results in downstream consequence Y.
  Example: demyelination results_in reduced conduction velocity.
* impaired_in - function or process impaired in disorder.
  Example: prefrontal cortex function impaired_in schizophrenia.
* symptom_of - clinical manifestation of disease.
  Example: resting tremor symptom_of Parkinson disease.
"""



MEANING_EXPL_GENERAL = MEANING_EXPL_qwen32_score55_exp_kg


# =========================================================
# System prompt template (keep your variable names)
# =========================================================

SYSTEM_CONTEXT_TEMPLATE = """You are a neuroscience-domain extractor building a neuroscience knowledge graph (KG) of triples <head, relation, tail>.
Given:
 • a sequence including neuroscience context
 • a list of heads (entities already extracted from the sequence)
Return for each head the relations chosen from the list below that could form a plausible KG triple and are supported by the sequence.

IMPORTANT CONSTRAINT:
- Please choose the closest match when multiple are possible, and only choose 1 relation per head.
- If none apply, return [].

Allowed relations:
###
{ALLOWED_RELATIONS}
###

Examples (head | relation | tail):
-------------
{EXAMPLES}
-------------

Meaning of relations:
{MEANING}

-------------   

The input format:
sequence

heads: [head1, head2, ...]

Output format:
{{
    "head 1": ["relation 1"],
    "head 2": [],
}}

##
Steps
Step 1: Understand Input
    Clearly understand the neuroscience context from the sequence.
    For each head, find explicit mentions in the text.
    Check if each head is explicitly linked to other concepts or relations.

Step 2: Use the list of allowed relations. Evaluate each head individually.
    Don't abuse "associated_with" — use it only if no more specific relation is supported.

Step 3: For each head, output exactly ONE relation (as a single-item list) or [].
##

Think concisely within <think></think>. Immediately after, output JSON."""


# Use the same assignment style you used
SYSTEM_CONTEXT = SYSTEM_CONTEXT_TEMPLATE.format(
    ALLOWED_RELATIONS=ALLOWED_RELATIONS_qwen32_score55_exp_kg,
    EXAMPLES=EXAMPLES_qwen32_score55_exp_kg,
    MEANING=MEANING_EXPL_qwen32_score55_exp_kg,
)


# =========================================================
# Example dialogs (neuroscience-specific; keep same var names)
# =========================================================

example_user_1 = """Input:
Sequence:
Activation of NMDA receptors on CA1 pyramidal neurons triggered calcium influx and was required for long-term potentiation (LTP) in the hippocampus. The hippocampus encodes episodic memory and is part of the limbic system.

heads: ["NMDA receptors", "calcium influx", "long-term potentiation (LTP)", "hippocampus", "episodic memory", "limbic system"]

Output:"""

example_assistant_1 = """{
    "NMDA receptors": ["required_for"],
    "calcium influx": [],
    "long-term potentiation (LTP)": [],
    "hippocampus": ["part_of"],
    "episodic memory": [],
    "limbic system": []
}"""

example_explanation_1 = """NMDA receptors are explicitly required for LTP -> required_for.
The text states hippocampus is part of limbic system -> part_of.
Other heads are not framed in a directional relation as heads here, so [] is acceptable under the constraints."""


# -----

example_user_2 = """Input:
Sequence:
Dopaminergic neurons in the ventral tegmental area (VTA) project to the nucleus accumbens and release dopamine during reward prediction.

heads: ["dopaminergic neurons", "ventral tegmental area", "nucleus accumbens", "dopamine", "reward prediction"]

Output:"""

example_assistant_2 = """{
    "dopaminergic neurons": ["releases"],
    "ventral tegmental area": ["projects_to"],
    "nucleus accumbens": [],
    "dopamine": ["mediates_signal_for"],
    "reward prediction": []
}"""

example_explanation_2 = """The sequence explicitly states dopaminergic neurons release dopamine -> releases.
It explicitly states VTA projects to nucleus accumbens -> projects_to.
Dopamine is described as mediating reward prediction signaling -> mediates_signal_for.
Other heads not directly expressed as relational sources under the constraint -> []."""


# -----

example_user_3 = """Input:
Sequence:
In Parkinson disease, degeneration of the substantia nigra is associated with dopaminergic neuron loss and produces resting tremor. Levodopa is used in Parkinson disease.

heads: ["Parkinson disease", "substantia nigra", "dopaminergic neuron loss", "resting tremor", "levodopa"]

Output:"""

example_assistant_3 = """{
    "Parkinson disease": [],
    "substantia nigra": ["impaired_in"],
    "dopaminergic neuron loss": [],
    "resting tremor": ["symptom_of"],
    "levodopa": []
}"""

example_explanation_3 = """Resting tremor is explicitly described as a symptom of Parkinson disease -> symptom_of.
Degeneration/impairment of substantia nigra is described in Parkinson disease; among allowed relations, impaired_in is the closest match -> impaired_in.
Levodopa is mentioned but there is no allowed treated_by relation, so [] is used rather than inventing a relation."""


# -----

example_user_4 = """Input:
Sequence:
Layer 2/3 pyramidal neurons are located in the prefrontal cortex. The prefrontal cortex is connected to the anterior cingulate cortex. Norepinephrine modulates attention.

heads: ["prefrontal cortex", "pyramidal neurons", "anterior cingulate cortex", "norepinephrine", "attention"]

Output:"""

example_assistant_4 = """{
    "prefrontal cortex": ["connected_to"],
    "pyramidal neurons": ["located_in"],
    "anterior cingulate cortex": [],
    "norepinephrine": ["modulates"],
    "attention": []
}"""

example_explanation_4 = """The sequence explicitly states PFC is connected to ACC -> connected_to.
It states pyramidal neurons are located in PFC -> located_in.
Norepinephrine modulates attention -> modulates.
Other heads are not used as relation sources under the single-relation constraint -> []."""


# -----

example_user_5 = """Input:
Sequence:
In the cerebral cortex, pyramidal neurons contain NMDA receptors. NMDA receptor activation activates calcium influx and is required for long-term potentiation.

heads: ["cerebral cortex", "pyramidal neurons", "NMDA receptors", "calcium influx", "long-term potentiation"]

Output:"""

example_assistant_5 = """{
    "cerebral cortex": ["contains"],
    "pyramidal neurons": ["contains"],
    "NMDA receptors": ["required_for"],
    "calcium influx": [],
    "long-term potentiation": []
}"""

example_explanation_5 = """Cerebral cortex contains pyramidal neurons -> contains.
Pyramidal neurons contain NMDA receptors -> contains.
NMDA receptors are explicitly required for long-term potentiation -> required_for.
Other heads are not expressed as relation sources under the one-relation constraint -> [] is acceptable."""


expanded_kg_example_user_4 = example_user_4
expanded_kg_assistant_4 = example_assistant_4
expanded_kg_explanation_4 = example_explanation_4

expanded_kg_example_user_5 = example_user_5
expanded_kg_assistant_5 = example_assistant_5
expanded_kg_explanation_5 = example_explanation_5


# =========================================================
# (Optional) Qwen14 placeholders to preserve structure if referenced
# =========================================================

qwen14_example_assistant_1 = example_assistant_1
qwen14_example_explanation_1 = example_explanation_1

qwen14_example_user_2 = example_user_2
qwen14_example_assistant_2 = example_assistant_2
qwen14_example_explanation_2 = example_explanation_2

qwen14_example_user_3 = example_user_3
qwen14_example_assistant_3 = example_assistant_3
qwen14_example_explanation_3 = example_explanation_3

qwen14_example_user_4 = example_user_4
qwen14_example_assistant_4 = example_assistant_4
qwen14_example_explanation_4 = example_explanation_4

qwen14_example_user_5 = example_user_5
qwen14_example_assistant_5 = example_assistant_5
qwen14_example_explanation_5 = example_explanation_5