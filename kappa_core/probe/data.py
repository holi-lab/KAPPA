from __future__ import annotations
from typing import List, Dict
import torch as t
import random
from torch.utils.data import Dataset

random.seed(42)

class ProbeItemDataset(Dataset):
    def __init__(self, activations:List, items: List[Dict], objective: str):
        self.activations = activations
        self.items = items
        self.objective = objective

        self.num_option = len(items[0]["options"])
        if self.objective == "error":
            self.labels = t.tensor(
                [
                    item["answer"] * self.num_option + item["pred"] 
                    for item in self.items
                ]
            )
        else:
            self.labels = t.tensor([item[objective] for item in self.items]) # score, pred, answer
        
        self.n_classes = self.num_option**2 if self.objective == "error" \
                    else self.num_option    if self.objective == "pred"  \
                    else self.num_option    if self.objective == "answer" \
                    else 2                  # for "score"
                         
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return idx, self.activations[idx], self.labels[idx]
