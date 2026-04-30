import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import json
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

# ==========================================
# 1. MODEL DEFINITIONS
# ==========================================

def load_negation_direction(vector_path):
    payload = torch.load(vector_path, map_location="cpu")
    if isinstance(payload, dict) and "W_dir" in payload:
        return payload["W_dir"].float()
    if isinstance(payload, dict) and "W_dir_list" in payload:
        return payload["W_dir_list"].float()  # [L, D]
    if torch.is_tensor(payload):
        return payload.float()
    raise ValueError(f"Unsupported negation vector format in {vector_path}")

class NegationSteeredCLIP(nn.Module):
    def __init__(self, base_clip_model, w_dir, alpha=0.13, multi_layer=False):
        super().__init__()
        self.base_model = base_clip_model
        self.alpha = alpha
        self.multi_layer = multi_layer
        
        if w_dir.dim() == 1:
            w_dir = w_dir.unsqueeze(0)
        self.w_dir = F.normalize(w_dir.float(), p=2, dim=-1)
        
        self.logit_scale = base_clip_model.logit_scale
        self.visual = base_clip_model.visual
        
        if multi_layer and self.w_dir.shape[0] > 1:
            self._setup_layer_hooks()
    
    def _setup_layer_hooks(self):
        text_transformer = self.base_model.transformer
        layers = text_transformer.resblocks
        self.layer_hooks = []
        
        for i, layer in enumerate(layers):
            if i < self.w_dir.shape[0]:
                def make_hook(layer_idx):
                    def hook_fn(module, input, output):
                        out = output[0] if isinstance(output, tuple) else output
                        if not hasattr(self, '_current_eos_indices'):
                            return output 
                            
                        eos_idx = self._current_eos_indices
                        batch_idx = torch.arange(out.shape[1])
                        
                        h_eos = out[eos_idx, batch_idx, :]
                        norm_h = torch.norm(h_eos, p=2, dim=-1, keepdim=True)
                        h_dir = h_eos / norm_h 
                        
                        w = self.w_dir[layer_idx].to(out.device)
                        w_dir = F.normalize(w, p=2, dim=-1)
                        if w_dir.dim() == 1:
                            w_dir = w_dir.unsqueeze(0) 
                            
                        interpolated_dir = (1 - self.alpha) * h_dir + self.alpha * w_dir
                        steered_dir = F.normalize(interpolated_dir, p=2, dim=-1)
                        steered_eos = steered_dir * norm_h
                        
                        out_steered = out.clone()
                        out_steered[eos_idx, batch_idx, :] = steered_eos.to(out.dtype)
                        
                        if isinstance(output, tuple):
                            return (out_steered,) + output[1:]
                        return out_steered
                return hook_fn
                
                hook = layer.register_forward_hook(make_hook(i))
                self.layer_hooks.append(hook)

    def encode_text(self, text_input):
        if isinstance(text_input, str):
            tokens = clip.tokenize([text_input]).to(next(self.base_model.parameters()).device)
        elif isinstance(text_input, (list, tuple)) and text_input and isinstance(text_input[0], str):
            tokens = clip.tokenize(list(text_input)).to(next(self.base_model.parameters()).device)
        else:
            tokens = text_input

        self._current_eos_indices = tokens.argmax(dim=-1)

        x = self.base_model.token_embedding(tokens).type(self.base_model.dtype)  
        x = x + self.base_model.positional_embedding.type(self.base_model.dtype)
        x = x.permute(1, 0, 2)  
        
        x = self.base_model.transformer(x)
        
        x = x.permute(1, 0, 2)  
        x = self.base_model.ln_final(x).type(self.base_model.dtype)
        
        x = x[torch.arange(x.shape[0]), self._current_eos_indices] 
        x = x @ self.base_model.text_projection  
        
        if not self.multi_layer:
            norm_h = torch.norm(x, p=2, dim=-1, keepdim=True)
            h_dir = x / norm_h
            
            w = self.w_dir[0].to(x.device)
            w_dir = F.normalize(w, p=2, dim=-1)
            if w_dir.dim() == 1:
                w_dir = w_dir.unsqueeze(0)
                
            interpolated_dir = (1 - self.alpha) * h_dir + self.alpha * w_dir
            steered_dir = F.normalize(interpolated_dir, p=2, dim=-1)
            x = steered_dir * norm_h
            
        delattr(self, '_current_eos_indices')
            
        return x

    def encode_image(self, image):
        return self.base_model.encode_image(image)
    
    def remove_hooks(self):
        for hook in self.layer_hooks:
            hook.remove()
        self.layer_hooks = []