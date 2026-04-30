"""
Feature cache management.
Handles disk-based caching of computed embeddings.
"""

import os
import json
import hashlib
import torch


class FeatureCache:
    """
    Manages caching of computed embeddings to disk.
    Uses MD5 hashing of model name, dataset, and config to create unique cache IDs.
    """
    def __init__(self, cache_dir="./embeddings_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, model_variant_name, dataset_name, config):
        """
        Generate unique cache path for embeddings.
        
        Args:
            model_variant_name: e.g., 'Baseline_CLIP_ViT_B32'
            dataset_name: e.g., 'NegationCLIP'
            config: dict of hyperparams affecting embeddings (e.g., {'layer': 4})
        
        Returns:
            str: Absolute path to cache file
        """
        config_str = json.dumps(config, sort_keys=True)
        unique_id = hashlib.md5(f"{model_variant_name}_{dataset_name}_{config_str}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{model_variant_name}_{unique_id}.pt")

    def save(self, data, path):
        """Save data to cache file"""
        torch.save(data, path)
        print(f"--- Features successfully cached at {path} ---")

    def load(self, path):
        """Load data from cache file if exists"""
        if os.path.exists(path):
            print(f"--- Found cached features: {path} ---")
            return torch.load(path)
        return None
