"""
Feature extraction from text using CLIP.
Supports layer-wise extraction from transformer.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from .cache import FeatureCache


def extract_and_cache(model, dataloader, tokenizer, model_variant_name, config, dataset="NegationCLIP", device="cuda"):
    """
    Extract text embeddings at specified layer and cache them.
    
    Args:
        model: CLIP model
        dataloader: DataLoader with 'pos_text' and 'neg_text' keys
        tokenizer: CLIP tokenizer
        model_variant_name: Name for cache identification
        config: dict with 'layer' key (-1 for final layer, 1-12 for intermediate layers)
        dataset: Dataset name for cache organization
        device: 'cuda' or 'cpu'
    
    Returns:
        dict: {"pos_text": Tensor, "neg_text": Tensor} of normalized embeddings
    """
    cache_manager = FeatureCache()
    cache_path = cache_manager.get_cache_path(model_variant_name, dataset, config)
    
    # Return cached if exists
    cached_data = cache_manager.load(cache_path)
    if cached_data:
        return cached_data

    pos_text_features, neg_text_features = [], []
    layer_idx = config.get('layer', -1)  # Default to final layer

    model.to(device)
    model.eval()
    
    def get_layer_features(tokens):
        """Extract features from specific transformer layer"""
        x = model.token_embedding(tokens).type(model.dtype)
        x = x + model.positional_embedding.type(model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Pass through transformer up to layer_idx
        num_layers = len(model.transformer.resblocks)
        target = layer_idx if layer_idx != -1 else num_layers
        
        for i in range(target):
            x = model.transformer.resblocks[i](x)
            
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = model.ln_final(x).type(model.dtype)

        # Extract [EOS] token representation
        state = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
        return state

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {model_variant_name} (Layer {layer_idx})"):
            p_tokens = tokenizer(batch['pos_text']).to(device)
            n_tokens = tokenizer(batch['neg_text']).to(device)

            if "DEO" in model_variant_name:
                # DEO models with custom encode_text method
                p_feat = model.encode_text(batch['pos_text'])
                n_feat = model.encode_text(batch['neg_text'])
            else:
                p_feat = get_layer_features(p_tokens)
                n_feat = get_layer_features(n_tokens)

            # Normalize to unit hypersphere
            p_feat = F.normalize(p_feat.float(), p=2, dim=-1)
            n_feat = F.normalize(n_feat.float(), p=2, dim=-1)

            pos_text_features.append(p_feat.cpu())
            neg_text_features.append(n_feat.cpu())

    data_to_cache = {
        "pos_text": torch.cat(pos_text_features),
        "neg_text": torch.cat(neg_text_features),
    }
    
    cache_manager.save(data_to_cache, cache_path)
    return data_to_cache
