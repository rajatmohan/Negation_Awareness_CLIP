"""
Binary negation classifier training.
Learns a linear direction that separates positive from negative embeddings.
Uses L-BFGS solver as per the paper (no bias term).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
import numpy as np


def train_binary_negation_classifier(z_pos, z_neg, config):
    """
    Train linear binary classifier using L-BFGS to separate positive vs negative embeddings.
    Follows paper methodology: no bias term, L-BFGS solver, max 1000 iterations.
    Returns normalized weight direction.
    
    Args:
        z_pos: Positive embeddings [N, D]
        z_neg: Negative embeddings [N, D]
        config: dict with keys: device, max_iter (default 1000)
    
    Returns:
        (W_dir, history) tuple
            W_dir: Normalized classifier weights [1, D]
            history: Training metrics dict
    """
    device = config.get('device', 'cuda')
    max_iter = config.get('max_iter', 1000)  # Per paper: max 1000 iterations
    
    # Prepare data on CPU for scipy
    z_pos = z_pos.float().cpu().numpy()
    z_neg = z_neg.float().cpu().numpy()
    
    X = np.concatenate([z_pos, z_neg], axis=0)
    y = np.concatenate([np.zeros(len(z_pos)), np.ones(len(z_neg))], axis=0)
    
    D = X.shape[1]  # Embedding dimension
    
    # Loss function: Binary cross-entropy with logits
    def loss_fn(w):
        """Binary cross-entropy loss with logits (no bias term)"""
        w_reshaped = w.reshape(1, -1)  # [1, D]
        logits = X @ w_reshaped.T  # [N, 1]
        logits = logits.squeeze()
        
        # Numerically stable BCEWithLogitsLoss
        # Loss = max(logits, 0) - logits * y + log(1 + exp(-abs(logits)))
        max_vals = np.maximum(logits, 0)
        loss = max_vals - logits * y + np.log(1 + np.exp(-np.abs(logits)))
        return loss.mean()
    
    def grad_fn(w):
        """Gradient of loss w.r.t. w"""
        w_reshaped = w.reshape(1, -1)
        logits = X @ w_reshaped.T
        logits = logits.squeeze()
        
        # sigmoid(logits)
        sig = 1.0 / (1.0 + np.exp(-logits))
        
        # dL/dw = X^T * (sig - y) / N
        grad = (X.T @ (sig - y)) / len(y)
        return grad.flatten()
    
    # Initialize weights
    w_init = np.random.randn(D) * 0.01
    
    # Run L-BFGS optimization
    print("Training binary classifier with L-BFGS (no bias term)...")
    result = minimize(
        loss_fn,
        w_init,
        method='L-BFGS-B',
        jac=grad_fn,
        options={'maxiter': max_iter, 'disp': False}
    )
    
    w_opt = result.x
    
    # Normalize weights to get direction vector
    w_norm = np.linalg.norm(w_opt)
    W_dir = w_opt / w_norm
    W_dir = torch.from_numpy(W_dir).float().unsqueeze(0)  # [1, D]
    
    # Compute metrics for history
    history = {
        'final_loss': loss_fn(w_opt),
        'iterations': result.nit,
        'success': result.success,
        'message': result.message,
        'weight_norm': w_norm
    }
    
    # Compute training accuracy
    logits = X @ w_opt.reshape(-1, 1)
    preds = (logits.squeeze() > 0).astype(float)
    train_acc = (preds == y).mean()
    history['train_acc'] = train_acc
    
    print(f"L-BFGS completed in {result.nit} iterations")
    print(f"Final Loss: {history['final_loss']:.5f}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Weight norm: {w_norm:.4f}")
    
    return W_dir, history
