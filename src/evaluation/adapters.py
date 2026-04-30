"""Model adapters for unified evaluation interface"""

import torch
import torch.nn.functional as F
import clip


class PairwiseModelAdapter:
    """
    Unified interface for encoding text and images across different models.
    Supports both tokenized (CLIP) and raw text input modes.
    """
    def __init__(self, name, model, device, text_mode="tokenized"):
        """
        Args:
            name: Model identifier
            model: Model with encode_text and encode_image methods
            device: 'cuda' or 'cpu'
            text_mode: 'tokenized' (CLIP-style) or 'raw' (DEO-style)
        """
        self.name = name
        self.model = model
        self.device = device
        self.text_mode = text_mode

    def encode_text(self, prompt):
        """Encode text to normalized embedding"""
        with torch.no_grad():
            if self.text_mode == "tokenized":
                tokens = clip.tokenize([prompt]).to(self.device)
                text_feat = self.model.encode_text(tokens).float()
            elif self.text_mode == "raw":
                text_feat = self.model.encode_text(prompt).float()
            else:
                raise ValueError(f"Unknown text mode: {self.text_mode}")

            return F.normalize(text_feat, p=2, dim=-1)

    def encode_image(self, image_tensor):
        """Encode image to normalized embedding"""
        with torch.no_grad():
            image_feat = self.model.encode_image(image_tensor).float()
            return F.normalize(image_feat, p=2, dim=-1)
