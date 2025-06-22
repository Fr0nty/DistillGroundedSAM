import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import warnings
import cv2
warnings.filterwarnings('ignore')

# Transformers and vision models
from transformers import AutoProcessor, AutoModel
import timm
from einops import rearrange, repeat

# Dataset utilities
from datasets import load_dataset

# GroundingDINO and SAM imports
try:
    import groundingdino
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    GROUNDINGDINO_AVAILABLE = True
    print(" GroundingDINO successfully imported!")
except ImportError as e:
    print(f"  GroundingDINO not available: {e}")
    print("Using enhanced CLIP fallback...")
    GROUNDINGDINO_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    print(" SAM successfully imported!")
except ImportError as e:
    print(f"  SAM not available: {e}")
    SAM_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ViTStudent(nn.Module):
    """
    Vision Transformer Student Model
    Learns to mimic GroundingDINO's visual representations
    """
    
    def __init__(self, 
                 image_size=224,
                 patch_size=16,
                 embed_dim=384,
                 depth=6,
                 num_heads=6,
                 num_classes=1000):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=depth
        )
        
        # Feature projection heads (for distillation)
        self.feature_projectors = nn.ModuleList([
            nn.Linear(embed_dim, 512),  # For matching teacher features
            nn.Linear(embed_dim, 768),  # Alternative projection
        ])
        
        # Classification head (optional)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input images [B, C, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing outputs and features
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        # Store intermediate features
        features = []
        
        # Transformer layers
        for i, layer in enumerate(self.transformer.layers):
            x = layer(x)
            if return_features:
                features.append(x.clone())
        
        # Extract class token and patch tokens
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        
        # Feature projections
        projected_features = []
        for projector in self.feature_projectors:
            projected_features.append(projector(cls_token))
        
        # Classification
        logits = self.classifier(cls_token)
        
        output = {
            'logits': logits,
            'cls_token': cls_token,
            'patch_tokens': patch_tokens,
            'projected_features': projected_features,
            'intermediate_features': features if return_features else None
        }
        
        return output


class MAEStudent(nn.Module):
    """
    Masked Autoencoder Student Model
    Learns teacher representations through masked reconstruction
    """
    
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 embed_dim=384,
                 depth=6,
                 num_heads=6,
                 decoder_embed_dim=192,
                 decoder_depth=4,
                 decoder_num_heads=3,
                 mask_ratio=0.75):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=depth
        )
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.randn(decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, decoder_embed_dim))
        
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=decoder_embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=decoder_depth
        )
        
        # Prediction head
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3)
        
        # Feature projectors for distillation
        self.feature_projectors = nn.ModuleList([
            nn.Linear(embed_dim, 512),
            nn.Linear(embed_dim, 768),
        ])
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def random_masking(self, x, mask_ratio):
        """
        Random masking following MAE
        
        Args:
            x: [B, N, D] where N = num_patches + 1 (including cls token)
            mask_ratio: ratio of patches to mask
            
        Returns:
            x_masked: visible patches
            mask: binary mask, 0 is keep, 1 is remove
            ids_restore: indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int((N - 1) * (1 - mask_ratio))  # -1 for cls token
        
        # Generate random indices (excluding cls token)
        noise = torch.rand(B, N-1, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep subset of patches + cls token
        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device), ids_keep + 1], dim=1)
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N-1], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x, return_features=False):
        """
        Forward pass with optional masking
        
        Args:
            x: Input images [B, C, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing outputs and features
        """
        B = x.shape[0]
        
        # Patch embedding
        x_patches = self.patch_embed(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x_patches = x_patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_full = torch.cat([cls_tokens, x_patches], dim=1)
        x_full = x_full + self.pos_embed
        
        # Masking (only during training or if specifically requested)
        if self.training:
            x_visible, mask, ids_restore = self.random_masking(x_full, self.mask_ratio)
        else:
            x_visible = x_full
            mask = torch.zeros(B, self.num_patches, device=x.device)
            ids_restore = None
        
        # Encoder
        features = []
        x_encoded = x_visible
        for layer in self.encoder.layers:
            x_encoded = layer(x_encoded)
            if return_features:
                features.append(x_encoded.clone())
        
        # Feature projections from cls token
        cls_token = x_encoded[:, 0]
        projected_features = []
        for projector in self.feature_projectors:
            projected_features.append(projector(cls_token))
        
        # Decoder (for reconstruction)
        if self.training or ids_restore is not None:
            x_decoded = self.decoder_embed(x_encoded)
            
            # Add mask tokens
            mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] + 1 - x_decoded.shape[1], 1)
            x_full_decoded = torch.cat([x_decoded[:, 1:, :], mask_tokens], dim=1)  # Remove cls token
            x_full_decoded = torch.gather(x_full_decoded, dim=1, 
                                        index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full_decoded.shape[2]))
            
            # Add cls token back and position embeddings
            cls_token_decoded = x_decoded[:, 0:1, :]
            x_full_decoded = torch.cat([cls_token_decoded, x_full_decoded], dim=1)
            x_full_decoded = x_full_decoded + self.decoder_pos_embed
            
            # Decoder transformer
            for layer in self.decoder.layers:
                x_full_decoded = layer(x_full_decoded)
            
            # Prediction
            reconstruction = self.decoder_pred(x_full_decoded[:, 1:, :])  # Remove cls token
        else:
            reconstruction = None
        
        output = {
            'reconstruction': reconstruction,
            'mask': mask,
            'cls_token': cls_token,
            'projected_features': projected_features,
            'intermediate_features': features if return_features else None,
            'encoded_features': x_encoded
        }
        
        return output