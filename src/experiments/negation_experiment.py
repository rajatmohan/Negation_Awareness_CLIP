"""
Main negation experiment: learn and evaluate negation direction.
Trains classifier, learns direction, evaluates on test set.
"""

import os
import json
import hashlib
import torch
import torch.nn.functional as F
from ..features import FeatureCache
from ..training import train_binary_negation_classifier, steer_embeddings


def run_paper_negation_experiment(config):
    """
    Complete negation experiment pipeline.
    
    Args:
        config: dict with keys:
            - dataset: dataset name
            - pos_variant, neg_variant: feature cache names
            - pos_config, neg_config: feature extraction configs
            - split_ratio: train/test split
            - val_split: train/val split for classifier
            - alpha: steering strength
            - lr, epochs, batch_size: classifier hyperparams
            - device, seed
    
    Returns:
        dict with W_dir, history, test_acc, baseline_sim, result_sim, gain
    """
    torch.manual_seed(config['seed'])
    cache_manager = FeatureCache()
    device = config['device']
    
    # Load features
    pos_path = cache_manager.get_cache_path(config['pos_variant'], config['dataset'], config['pos_config'])
    neg_path = cache_manager.get_cache_path(config['neg_variant'], config['dataset'], config['neg_config'])
    pos_data, neg_data = cache_manager.load(pos_path), cache_manager.load(neg_path)
    
    if pos_data is None or neg_data is None:
        raise FileNotFoundError(f"Could not load features from {pos_path} or {neg_path}")
    
    z_pos, z_neg = pos_data['pos_text'], neg_data['neg_text']

    # Train/test split
    indices = torch.randperm(z_pos.size(0))
    train_size = int(config['split_ratio'] * z_pos.size(0))
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    # Train classifier
    W_dir, history = train_binary_negation_classifier(
        z_pos[train_idx], z_neg[train_idx], config
    )

    # Evaluate on test set
    with torch.no_grad():
        test_p = z_pos[test_idx].to(device).float()
        test_n = z_neg[test_idx].to(device).float()
        
        # Test accuracy
        test_X = torch.cat([test_p, test_n], dim=0)
        test_y = torch.cat([torch.zeros(len(test_p)), torch.ones(len(test_n))], dim=0).to(device)
        test_logits = (test_X @ W_dir.T).squeeze()
        test_preds = (test_logits > 0).float()
        test_acc = (test_preds == test_y).sum().item() / len(test_y)

        # Steering evaluation
        steered_p = steer_embeddings(test_p, W_dir, config['alpha'])
        
        # Similarities
        p_n, n_n, s_p_n = F.normalize(test_p, p=2), F.normalize(test_n, p=2), F.normalize(steered_p, p=2)
        baseline_sim = torch.cosine_similarity(p_n, n_n).mean().item()
        final_sim = torch.cosine_similarity(s_p_n, n_n).mean().item()

    # Report
    print(f"\n" + "="*50)
    print(f"NEGATION EXPERIMENT: {config['pos_variant']}")
    print(f"="*50)
    print(f"Classifier Training Acc:   {history['train_acc'][-1]:.2%}")
    print(f"Classifier Validation Acc: {history['val_acc'][-1]:.2%}")
    print(f"Classifier Test Acc:       {test_acc:.2%}")
    print(f"-"*50)
    print(f"Baseline Cosine Sim:       {baseline_sim:.4f}")
    print(f"Steered Cosine Sim:        {final_sim:.4f}")
    print(f"Total Gain:                {final_sim - baseline_sim:.4f}")
    print(f"="*50)

    output = {
        "W_dir": W_dir.cpu(),
        "history": history,
        "test_acc": test_acc,
        "baseline_sim": baseline_sim,
        "result_sim": final_sim,
        "gain": final_sim - baseline_sim
    }

    # Save results
    results_dir = "learned_vectors"
    os.makedirs(results_dir, exist_ok=True) 
    
    config_dict = {
        "dataset": config['dataset'],
        "pos_variant": config['pos_variant'],
        "neg_variant": config['neg_variant'],
        "pos_config": config['pos_config'],
        "neg_config": config['neg_config'],
    }

    config_str = json.dumps(config_dict, sort_keys=True)
    hash_id = hashlib.sha1(config_str.encode()).hexdigest()[:8]
    safe_name = f"{config['dataset']}_{config['pos_variant']}_{config['neg_variant']}".replace("/", "-")

    save_filename = f"{safe_name}_{hash_id}_negation_vector.pt"
    save_path = os.path.join(results_dir, save_filename)

    torch.save(output, save_path)

    with open(os.path.join(results_dir, f"{safe_name}_{hash_id}_config.json"), "w") as f:
        f.write(config_str)

    print(f"Learned vector saved to: {save_path}")

    return output
