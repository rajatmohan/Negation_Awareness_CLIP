"""Training utilities and helper functions"""

import torch
import torch.nn.functional as F


def steer_embeddings(h_l, W_dir, alpha):
    """
    Apply negation steering to embeddings.
    Formula: h_steered = (1 - α) * h + α * W_dir * ||h||
    
    Args:
        h_l: Original embeddings [N, D]
        W_dir: Negation direction [1, D]
        alpha: Steering strength (0-1)
    
    Returns:
        Steered embeddings [N, D]
    """
    norm_h = torch.norm(h_l, p=2, dim=-1, keepdim=True)
    h_steered = (1 - alpha) * h_l + alpha * W_dir * norm_h
    return h_steered
