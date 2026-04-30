"""
Text steering evaluation: measure negation effect on text similarity.
Tests how steering affects similarity between positive and negative texts.
"""

import torch
import torch.nn.functional as F
from ..training import steer_embeddings


def evaluate_negation_steering_on_text(z_pos, z_neg, W_dir, alpha_values=[0.0, 0.3, 0.5, 0.7, 1.0], device="cuda"):
    """
    Evaluate negation steering on text embeddings.
    Measures how much negative texts are pushed away from positive texts.
    
    Args:
        z_pos: Positive embeddings [N, D]
        z_neg: Negative embeddings [N, D]
        W_dir: Negation direction [1, D]
        alpha_values: List of steering strengths to test
        device: 'cuda' or 'cpu'
    
    Returns:
        dict with alpha values and corresponding metrics
    """
    z_pos = z_pos.float().to(device)
    z_neg = z_neg.float().to(device)
    W_dir = W_dir.float().to(device)
    
    results = {"alpha": [], "baseline_sim": [], "steered_sim": [], "improvement": []}
    
    # Baseline (no steering)
    z_pos_norm = F.normalize(z_pos, p=2, dim=-1)
    z_neg_norm = F.normalize(z_neg, p=2, dim=-1)
    baseline_sim = torch.cosine_similarity(z_pos_norm, z_neg_norm).mean().item()
    
    print("\n=== Text Steering Evaluation ===")
    print(f"Baseline (no steering)  | Sim(pos,neg) = {baseline_sim:.4f}\n")
    
    for alpha in alpha_values:
        z_pos_steered = steer_embeddings(z_pos, W_dir, alpha)
        z_pos_steered_norm = F.normalize(z_pos_steered, p=2, dim=-1)
        steered_sim = torch.cosine_similarity(z_pos_steered_norm, z_neg_norm).mean().item()
        improvement = baseline_sim - steered_sim  # Higher = better (lower similarity)
        
        results["alpha"].append(alpha)
        results["baseline_sim"].append(baseline_sim)
        results["steered_sim"].append(steered_sim)
        results["improvement"].append(improvement)
        
        print(f"Alpha = {alpha:.2f}  | Sim(pos,neg) = {steered_sim:.4f} | Gap Improvement = {improvement:+.4f}")
    
    print("="*50)
    return results
