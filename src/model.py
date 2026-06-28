"""
Model architectures.
Owner: Le Nguyen Khoi, Pham Viet Truong
"""

import torch
import torch.nn as nn


class QueryClassifier(nn.Module):
    """
    Lightweight MLP for query type and complexity classification.
    Input:  sentence embedding (embed_dim,)
    Output: (type_logits [4], complexity_logits [3])

    Query types:   0=text-only, 1=ocr, 2=asr, 3=hybrid
    Complexity:    0=low, 1=medium, 2=high

    TODO (Le Nguyen Khoi):
    - Design and implement the network layers
    - Keep inference time <2ms on CPU (this feeds the dispatcher)
    """

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        # TODO: define layers
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: implement forward pass
        # return type_logits, complexity_logits
        raise NotImplementedError


class MultiTaskLoss(nn.Module):
    """
    Combined CrossEntropy loss for both classifier heads.

    TODO (Pham Viet Truong):
    - Implement weighted combination of type loss and complexity loss
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, type_logits, comp_logits, type_labels, comp_labels):
        raise NotImplementedError
