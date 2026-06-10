import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Union, Dict, Optional
from transformers import BertConfig, DataCollatorForLanguageModeling

class GraphMertConfig(BertConfig):
    model_type = "graphmert"
    def __init__(
        self, 
        root_nodes=512, 
        max_nodes=4608, 
        num_relationships=43, 
        graph_types=None, 
        pretrained_emb_dim=0,
        mlm_sbo=False,
        exp_mask_base=0.0,
        relation_emb_dropout=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root_nodes = root_nodes
        self.max_nodes = max_nodes
        self.num_relationships = num_relationships
        self.graph_types = graph_types if graph_types is not None else []
        self.pretrained_emb_dim = pretrained_emb_dim
        self.mlm_sbo = mlm_sbo
        self.exp_mask_base = exp_mask_base
        self.relation_emb_dropout = relation_emb_dropout

@dataclass
class GraphMertDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    config: GraphMertConfig = None
    graph_types: List[str] = None
    process_arch_tensors: bool = True
    on_the_fly_processing: bool = False
    mlm_sbo: bool = False
    mlm_on_leaves_probability: float = 0.15
    subword_token_start: str = "##"

    def preprocess_items(self, items):
        if 'input_nodes' in items and isinstance(items['input_nodes'][0], (int, np.integer)):
            return items

        input_ids = items['input_ids']
        leaf_node_ids = items.get('leaf_node_ids')
        bsz = len(input_ids)
        flat_inputs = []
        
        for i in range(bsz):
            roots = list(input_ids[i])
            if len(roots) < self.config.root_nodes:
                roots += [self.tokenizer.pad_token_id] * (self.config.root_nodes - len(roots))
            else:
                roots = roots[:self.config.root_nodes]
                
            if leaf_node_ids is not None and len(leaf_node_ids[i]) > 0:
                leaves = leaf_node_ids[i]
                if isinstance(leaves[0], (list, np.ndarray)):
                    flat_leaves = [item for sublist in leaves for item in sublist]
                else:
                    flat_leaves = list(leaves)
            else:
                flat_leaves = []
            
            total_leaf_slots = self.config.max_nodes - self.config.root_nodes
            if len(flat_leaves) < total_leaf_slots:
                flat_leaves += [self.tokenizer.pad_token_id] * (total_leaf_slots - len(flat_leaves))
            else:
                flat_leaves = flat_leaves[:total_leaf_slots]
            
            flat_inputs.append(roots + flat_leaves)

        items['input_nodes'] = flat_inputs
        return items

    def get_start_indices(self, items):
        bsz = len(items['input_nodes'])
        items['start_indices'] = [[0] * self.config.root_nodes for _ in range(bsz)]
        return items
