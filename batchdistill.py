import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from transformers import Dinov2Model, Dinov2Config
import math
from typing import Dict, Tuple, Optional
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StudentViT(nn.Module):
    """Lightweight ViT student model"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 0,  # 0 for feature extraction
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        
        # Use timm's ViT implementation for the student
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        
        self.embed_dim = embed_dim
        
        # Projection head for distillation
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x: torch.Tensor, return_features: bool = True) -> Dict[str, torch.Tensor]:
        # Get features from ViT backbone
        features = self.vit.forward_features(x)
        
        # Extract CLS token (first token)
        cls_token = features[:, 0]  # [B, embed_dim]
        
        # Project features for distillation
        projected_features = self.projection_head(cls_token)
        
        outputs = {
            'features': cls_token,
            'projected_features': projected_features,
        }
        
        if return_features:
            outputs['patch_features'] = features[:, 1:]  # Exclude CLS token
            
        return outputs

class DINOv2Teacher(nn.Module):
    """DINOv2 teacher model wrapper"""
    
    def __init__(self, model_name: str = "facebook/dinov2-small"):
        super().__init__()
        
        self.dinov2 = Dinov2Model.from_pretrained(model_name)
        self.dinov2.eval()  # Always in eval mode
        
        # Freeze teacher parameters
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.dinov2.config.hidden_size
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.dinov2(x, output_hidden_states=True)
        
        # Get CLS token from last layer
        cls_token = outputs.last_hidden_state[:, 0]  # [B, embed_dim]
        
        return {
            'features': cls_token,
            'patch_features': outputs.last_hidden_state[:, 1:],  # Exclude CLS token
            'hidden_states': outputs.hidden_states
        }

class DistillationLoss(nn.Module):
    """Combined distillation loss with multiple components"""
    
    def __init__(
        self,
        feature_loss_weight: float = 1.0,
        attention_loss_weight: float = 0.5,
        temperature: float = 4.0,
    ):
        super().__init__()
        
        self.feature_loss_weight = feature_loss_weight
        self.attention_loss_weight = attention_loss_weight
        self.temperature = temperature
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
    def feature_distillation_loss(
        self, 
        student_features: torch.Tensor, 
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss between normalized features"""
        
        # Normalize features
        student_norm = F.normalize(student_features, dim=-1)
        teacher_norm = F.normalize(teacher_features, dim=-1)
        
        return self.mse_loss(student_norm, teacher_norm)
    
    def cosine_similarity_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Cosine similarity loss"""
        
        target = torch.ones(student_features.size(0)).to(student_features.device)
        return self.cosine_loss(student_features, teacher_features, target)
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        
        losses = {}
        
        # Feature distillation loss
        feature_loss = self.feature_distillation_loss(
            student_outputs['projected_features'],
            teacher_outputs['features']
        )
        losses['feature_loss'] = feature_loss
        
        # Cosine similarity loss
        cosine_loss = self.cosine_similarity_loss(
            student_outputs['features'],
            teacher_outputs['features']
        )
        losses['cosine_loss'] = cosine_loss
        
        # Total loss
        total_loss = (
            self.feature_loss_weight * feature_loss +
            0.5 * cosine_loss
        )
        
        losses['total_loss'] = total_loss
        
        return losses

class DINOv2Distiller:
    """Main distillation trainer"""
    
    def __init__(
        self,
        student_config: Dict,
        teacher_model_name: str = "facebook/dinov2-small",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        temperature: float = 4.0,
    ):
        self.device = device
        
        # Initialize models
        self.student = StudentViT(**student_config).to(device)
        self.teacher = DINOv2Teacher(teacher_model_name).to(device)
        
        # Initialize loss
        self.criterion = DistillationLoss(temperature=temperature).to(device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler (will be set in train method)
        self.scheduler = None
        
        logger.info(f"Student model parameters: {sum(p.numel() for p in self.student.parameters()):,}")
        logger.info(f"Teacher model parameters: {sum(p.numel() for p in self.teacher.parameters()):,}")
        
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.student.train()
        
        total_loss = 0.0
        total_feature_loss = 0.0
        total_cosine_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                teacher_outputs = self.teacher(images)
                
            student_outputs = self.student(images)
            
            # Compute losses
            losses = self.criterion(student_outputs, teacher_outputs)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses['total_loss'].item()
            total_feature_loss += losses['feature_loss'].item()
            total_cosine_loss += losses['cosine_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Feature': f"{losses['feature_loss'].item():.4f}",
                'Cosine': f"{losses['cosine_loss'].item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
            
        return {
            'total_loss': total_loss / num_batches,
            'feature_loss': total_feature_loss / num_batches,
            'cosine_loss': total_cosine_loss / num_batches,
        }
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate the model"""
        
        self.student.eval()
        
        total_loss = 0.0
        total_feature_loss = 0.0
        total_cosine_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                
                teacher_outputs = self.teacher(images)
                student_outputs = self.student(images)
                
                losses = self.criterion(student_outputs, teacher_outputs)
                
                total_loss += losses['total_loss'].item()
                total_feature_loss += losses['feature_loss'].item()
                total_cosine_loss += losses['cosine_loss'].item()
                num_batches += 1
                
        return {
            'total_loss': total_loss / num_batches,
            'feature_loss': total_feature_loss / num_batches,
            'cosine_loss': total_cosine_loss / num_batches,
        }
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        num_epochs: int = 100,
        save_path: str = "dinov2_student.pth"
    ):
        """Full training loop"""
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            
            logger.info(
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Feature: {train_metrics['feature_loss']:.4f}, "
                f"Cosine: {train_metrics['cosine_loss']:.4f}"
            )
            
            # Validation
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                
                logger.info(
                    f"Val Loss: {val_metrics['total_loss']:.4f}, "
                    f"Feature: {val_metrics['feature_loss']:.4f}, "
                    f"Cosine: {val_metrics['cosine_loss']:.4f}"
                )
                
                # Save best model
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    self.save_model(save_path)
                    logger.info(f"Saved best model to {save_path}")
            else:
                # Save model every 10 epochs if no validation
                if (epoch + 1) % 10 == 0:
                    self.save_model(f"{save_path}_epoch_{epoch + 1}.pth")
                    
            print("-" * 50)
    
    def save_model(self, path: str):
        """Save the student model"""
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load the student model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Usage example
def main():
    # Your existing dataloader
    from data.hf_cifar import get_cifar_dataloader  # Import your dataloader
    
    # Get dataloaders
    train_loader = get_cifar_dataloader(
        split="train[:80%]",
        batch_size=32,
        shuffle=True
    )
    
    val_loader = get_cifar_dataloader(
        split="train[80%:]",
        batch_size=32,
        shuffle=False
    )
    
    # Student model configuration
    student_config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 384,  # Smaller than DINOv2
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.0,
    }
    
    # Initialize distiller
    distiller = DINOv2Distiller(
        student_config=student_config,
        teacher_model_name="facebook/dinov2-small",  # or dinov2-base
        learning_rate=1e-4,
        weight_decay=0.05,
        temperature=4.0
    )
    
    # Train the model
    distiller.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=100,
        save_path="dinov2_student_cifar10.pth"
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()