"""CLIP model wrapper: turns images and text into L2-normalized embeddings.

Uses vision_model/text_model + projection layers directly (same approach as Week 4),
which avoids return-type differences of get_*_features across transformers versions.
"""
import logging
from typing import List, Sequence, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class CLIPEncoder:
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s", model_name, self.device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @property
    def dim(self) -> int:
        return self.model.config.projection_dim

    @torch.no_grad()
    def encode_text(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        """Return array of shape (n, dim), L2-normalized."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(text=list(texts), return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        out = self.model.text_model(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"])
        feats = self.model.text_projection(out.pooler_output)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """Return array of shape (n, dim), L2-normalized."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        feats = self.model.visual_projection(out.pooler_output)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)