"""
DEO (Dynamic Embedding Optimization) Model.
Enhances CLIP embeddings using LLM decompositions and iterative optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from clip.simple_tokenizer import SimpleTokenizer


class DEOModel(nn.Module):
    """
    Dynamic Embedding Optimization for CLIP.
    Optimizes embeddings toward decomposed positive/negative intents.
    """
    def __init__(self, clip_model, extractor, config, device="cuda"):
        """
        Args:
            clip_model: Base CLIP model
            extractor: SubQueryExtractor for LLM decomposition
            config: dict with keys: lr, steps, pos_weight, neg_weight, reg_weight
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        self.base_model = clip_model
        self.extractor = extractor
        self.device = device
        self.tokenizer = SimpleTokenizer()
        
        self.lr = config.get('lr', 0.001)
        self.steps = config.get('steps', 50)
        self.pos_weight = config.get('pos_weight', 0.4)
        self.neg_weight = config.get('neg_weight', 0.4)
        self.reg_weight = config.get('reg_weight', 1.0)
        
        self.logit_scale = clip_model.logit_scale
        self.visual = clip_model.visual

    def _get_emb(self, texts):
        """Get CLIP embeddings for texts"""
        if not texts or len(texts) == 0:
            return None
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            return F.normalize(self.base_model.encode_text(tokens).float(), p=2, dim=-1)

    def encode_text(self, raw_text=None):
        """
        Optimize text embeddings using DEO.
        
        Args:
            raw_text: str or list of str
        
        Returns:
            Optimized embeddings [B, D] on unit hypersphere
        """
        # Ensure raw_text is a list
        if isinstance(raw_text, str):
            raw_text = [raw_text]

        # Get LLM decompositions
        batch_meta = self.extractor.get_decomposition_batch(raw_text)
        
        # Get original CLIP embeddings and anchors
        with torch.no_grad():
            orig_embs = self._get_emb(raw_text) 
            pos_anchors = [self._get_emb(m.get('positives', [])) for m in batch_meta]
            neg_anchors = [self._get_emb(m.get('negatives', [])) for m in batch_meta]

        # Optimization setup
        updated_embs = orig_embs.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([updated_embs], lr=self.lr)

        # Iterative geometric optimization
        with torch.enable_grad():
            for _ in range(self.steps):
                optimizer.zero_grad()
                
                batch_loss = 0
                for i in range(len(raw_text)):
                    # Regularization: stay close to original
                    dev_loss = torch.norm(updated_embs[i] - orig_embs[i])
                    
                    # Positive pull: attract to expanded intents
                    p_loss = 0
                    if pos_anchors[i] is not None:
                        p_loss = torch.norm(updated_embs[i] - pos_anchors[i], dim=1).mean()
                    
                    # Negative push: repel from negated intents
                    n_loss = 0
                    if neg_anchors[i] is not None:
                        n_loss = torch.norm(updated_embs[i] - neg_anchors[i], dim=1).mean()

                    batch_loss += (self.reg_weight * dev_loss + 
                                   self.pos_weight * p_loss - 
                                   self.neg_weight * n_loss)

                # Backward pass
                (batch_loss / len(raw_text)).backward()
                optimizer.step()

                # Constraint: project back onto unit hypersphere
                with torch.no_grad():
                    updated_embs.data = F.normalize(updated_embs.data, p=2, dim=-1)

        return updated_embs.detach()

    def encode_image(self, image):
        """Encode image through base CLIP"""
        return self.base_model.encode_image(image)

    def forward(self, image, text_tokens, raw_text=None):
        """Forward pass for image-text matching"""
        image_features = F.normalize(self.encode_image(image), p=2, dim=-1)
        text_features = F.normalize(self.encode_text(raw_text) if raw_text else self.base_model.encode_text(text_tokens), p=2, dim=-1)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
