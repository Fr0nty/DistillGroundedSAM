import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from transformers import Dinov2Model, Dinov2Config
import math
from typing import Dict, Tuple, Optional, List
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionHook:
    """Hook to capture attention weights from transformer blocks"""
    
    def __init__(self):
        self.attention_weights = []
        
    def hook_fn(self, module, input, output):
        # For timm ViT: output is (attention_output, attention_weights)
        if isinstance(output, tuple) and len(output) >= 2:
            self.attention_weights.append(output[1])  # attention weights
        
    def clear(self):
        self.attention_weights = []

class StudentViT(nn.Module):
    """Enhanced ViT student model with attention and intermediate feature extraction"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        distill_layers: List[int] = None,  # Which layers to distill
    ):
        super().__init__()
        
        # Use timm's ViT implementation
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
        self.depth = depth
        self.num_heads = num_heads
        
        # Default to middle layers for distillation
        if distill_layers is None:
            self.distill_layers = [depth//4, depth//2, 3*depth//4]
        else:
            self.distill_layers = distill_layers
            
        # Projection head for feature distillation
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Projection heads for intermediate layer distillation
        self.intermediate_projections = nn.ModuleDict()
        for layer_idx in self.distill_layers:
            self.intermediate_projections[str(layer_idx)] = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
        
        # Setup attention hooks
        self.attention_hook = AttentionHook()
        self._register_attention_hooks()
        
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights"""
        for i, block in enumerate(self.vit.blocks):
            # Hook into the attention module
            if hasattr(block, 'attn'):
                block.attn.register_forward_hook(self.attention_hook.hook_fn)
    
    def forward(self, x: torch.Tensor, return_features: bool = True) -> Dict[str, torch.Tensor]:
        # Clear previous attention weights
        self.attention_hook.clear()
        
        # Get patch embeddings
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        
        # Store intermediate features
        intermediate_features = {}
        
        # Forward through transformer blocks
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            
            # Store features from specified layers
            if i in self.distill_layers:
                # Extract CLS token and project
                cls_features = x[:, 0]  # [B, embed_dim]
                projected = self.intermediate_projections[str(i)](cls_features)
                intermediate_features[f'layer_{i}'] = {
                    'features': cls_features,
                    'projected_features': projected
                }
        
        # Final processing
        x = self.vit.norm(x)
        
        # Extract final CLS token
        cls_token = x[:, 0]  # [B, embed_dim]
        projected_features = self.projection_head(cls_token)
        
        outputs = {
            'features': cls_token,
            'projected_features': projected_features,
            'intermediate_features': intermediate_features,
            'attention_weights': self.attention_hook.attention_weights.copy()
        }
        
        if return_features:
            outputs['patch_features'] = x[:, 1:]  # Exclude CLS token
            
        return outputs

class DINOv2Teacher(nn.Module):
    """Enhanced DINOv2 teacher model with attention and intermediate feature extraction"""
    
    def __init__(self, model_name: str = "facebook/dinov2-small", distill_layers: List[int] = None):
        super().__init__()
        
        self.dinov2 = Dinov2Model.from_pretrained(model_name, output_attentions=True)
        self.dinov2.eval()
        
        # Freeze teacher parameters
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.dinov2.config.hidden_size
        self.num_layers = self.dinov2.config.num_hidden_layers
        
        # Default to corresponding layers for distillation
        if distill_layers is None:
            self.distill_layers = [self.num_layers//4, self.num_layers//2, 3*self.num_layers//4]
        else:
            self.distill_layers = distill_layers
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.dinov2(x, output_hidden_states=True, output_attentions=True)
        
        # Get final CLS token
        cls_token = outputs.last_hidden_state[:, 0]  # [B, embed_dim]
        
        # Extract intermediate features from specified layers
        intermediate_features = {}
        for layer_idx in self.distill_layers:
            if layer_idx < len(outputs.hidden_states):
                layer_output = outputs.hidden_states[layer_idx]
                cls_features = layer_output[:, 0]  # CLS token from this layer
                intermediate_features[f'layer_{layer_idx}'] = {
                    'features': cls_features,
                }
        
        # Process attention weights (average across heads)
        processed_attentions = []
        if outputs.attentions is not None:
            for attn in outputs.attentions:
                # attn shape: [B, num_heads, seq_len, seq_len]
                # Average across heads: [B, seq_len, seq_len]
                avg_attn = attn.mean(dim=1)
                processed_attentions.append(avg_attn)
        
        return {
            'features': cls_token,
            'patch_features': outputs.last_hidden_state[:, 1:],
            'hidden_states': outputs.hidden_states,
            'intermediate_features': intermediate_features,
            'attention_weights': processed_attentions
        }

class EnhancedDistillationLoss(nn.Module):
    """Enhanced distillation loss with attention transfer and intermediate layer distillation"""
    
    def __init__(
        self,
        feature_loss_weight: float = 1.0,
        attention_loss_weight: float = 0.5,
        intermediate_loss_weight: float = 0.3,
        temperature: float = 4.0,
        attention_temperature: float = 3.0,
    ):
        super().__init__()
        
        self.feature_loss_weight = feature_loss_weight
        self.attention_loss_weight = attention_loss_weight
        self.intermediate_loss_weight = intermediate_loss_weight
        self.temperature = temperature
        self.attention_temperature = attention_temperature
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        
    def feature_distillation_loss(
        self, 
        student_features: torch.Tensor, 
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss between normalized features"""
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
    
    def attention_transfer_loss(
        self,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor]
    ) -> torch.Tensor:
        """Attention transfer loss using KL divergence"""
        if not student_attentions or not teacher_attentions:
            return torch.tensor(0.0, device=student_attentions[0].device if student_attentions else teacher_attentions[0].device)
        
        total_loss = 0.0
        num_layers = min(len(student_attentions), len(teacher_attentions))
        
        for i in range(num_layers):
            student_attn = student_attentions[i]
            teacher_attn = teacher_attentions[i]
            
            # Handle different numbers of heads by averaging student attention across heads
            if len(student_attn.shape) == 4:  # [B, heads, seq, seq]
                student_attn = student_attn.mean(dim=1)  # [B, seq, seq]
            
            # Ensure same sequence length by interpolating if needed
            if student_attn.shape[-1] != teacher_attn.shape[-1]:
                # Simple truncation/padding for now - could use interpolation
                min_seq = min(student_attn.shape[-1], teacher_attn.shape[-1])
                student_attn = student_attn[:, :min_seq, :min_seq]
                teacher_attn = teacher_attn[:, :min_seq, :min_seq]
            
            # Apply temperature and compute KL divergence
            student_flat = (student_attn / self.attention_temperature).flatten(start_dim=1)
            teacher_flat = (teacher_attn / self.attention_temperature).flatten(start_dim=1)
            
            student_soft = F.log_softmax(student_flat, dim=1)
            teacher_soft = F.softmax(teacher_flat, dim=1)
            
            layer_loss = self.kl_div_loss(student_soft, teacher_soft)
            total_loss += layer_loss
            
        return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)
    
    def intermediate_layer_loss(
        self,
        student_intermediate: Dict[str, Dict[str, torch.Tensor]],
        teacher_intermediate: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Loss for intermediate layer features"""
        total_loss = 0.0
        num_layers = 0
        
        for layer_key in student_intermediate:
            if layer_key in teacher_intermediate:
                student_feat = student_intermediate[layer_key]['projected_features']
                teacher_feat = teacher_intermediate[layer_key]['features']
                
                # Normalize features
                student_norm = F.normalize(student_feat, dim=-1)
                teacher_norm = F.normalize(teacher_feat, dim=-1)
                
                layer_loss = self.mse_loss(student_norm, teacher_norm)
                total_loss += layer_loss
                num_layers += 1
        
        return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        
        losses = {}
        
        # Feature distillation loss (final layer)
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
        
        # Attention transfer loss
        attention_loss = self.attention_transfer_loss(
            student_outputs.get('attention_weights', []),
            teacher_outputs.get('attention_weights', [])
        )
        losses['attention_loss'] = attention_loss
        
        # Intermediate layer distillation loss
        intermediate_loss = self.intermediate_layer_loss(
            student_outputs.get('intermediate_features', {}),
            teacher_outputs.get('intermediate_features', {})
        )
        losses['intermediate_loss'] = intermediate_loss
        
        # Total loss
        total_loss = (
            self.feature_loss_weight * feature_loss +
            0.5 * cosine_loss +
            self.attention_loss_weight * attention_loss +
            self.intermediate_loss_weight * intermediate_loss
        )
        
        losses['total_loss'] = total_loss
        
        return losses

class EnhancedDINOv2Distiller:
    """Enhanced distillation trainer with attention transfer and intermediate layer distillation"""
    
    def __init__(
        self,
        student_config: Dict,
        teacher_model_name: str = "facebook/dinov2-small",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        temperature: float = 4.0,
        attention_temperature: float = 3.0,
        distill_layers: List[int] = None,
    ):
        self.device = device
        
        # Add distill_layers to student config
        if distill_layers is not None:
            student_config['distill_layers'] = distill_layers
        
        # Initialize models
        self.student = StudentViT(**student_config).to(device)
        self.teacher = DINOv2Teacher(teacher_model_name, distill_layers).to(device)
        
        # Initialize enhanced loss
        self.criterion = EnhancedDistillationLoss(
            temperature=temperature,
            attention_temperature=attention_temperature
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = None
        
        logger.info(f"Student model parameters: {sum(p.numel() for p in self.student.parameters()):,}")
        logger.info(f"Teacher model parameters: {sum(p.numel() for p in self.teacher.parameters()):,}")
        
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch with enhanced losses"""
        
        self.student.train()
        
        total_loss = 0.0
        total_feature_loss = 0.0
        total_cosine_loss = 0.0
        total_attention_loss = 0.0
        total_intermediate_loss = 0.0
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
            total_attention_loss += losses['attention_loss'].item()
            total_intermediate_loss += losses['intermediate_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Feat': f"{losses['feature_loss'].item():.4f}",
                'Attn': f"{losses['attention_loss'].item():.4f}",
                'Inter': f"{losses['intermediate_loss'].item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
            
        return {
            'total_loss': total_loss / num_batches,
            'feature_loss': total_feature_loss / num_batches,
            'cosine_loss': total_cosine_loss / num_batches,
            'attention_loss': total_attention_loss / num_batches,
            'intermediate_loss': total_intermediate_loss / num_batches,
        }
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate the model with enhanced metrics"""
        
        self.student.eval()
        
        total_loss = 0.0
        total_feature_loss = 0.0
        total_cosine_loss = 0.0
        total_attention_loss = 0.0
        total_intermediate_loss = 0.0
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
                total_attention_loss += losses['attention_loss'].item()
                total_intermediate_loss += losses['intermediate_loss'].item()
                num_batches += 1
                
        return {
            'total_loss': total_loss / num_batches,
            'feature_loss': total_feature_loss / num_batches,
            'cosine_loss': total_cosine_loss / num_batches,
            'attention_loss': total_attention_loss / num_batches,
            'intermediate_loss': total_intermediate_loss / num_batches,
        }
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        num_epochs: int = 100,
        save_path: str = "enhanced_dinov2_student.pth"
    ):
        """Full training loop with enhanced logging"""
        
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
                f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                f"Feature: {train_metrics['feature_loss']:.4f}, "
                f"Attention: {train_metrics['attention_loss']:.4f}, "
                f"Intermediate: {train_metrics['intermediate_loss']:.4f}"
            )
            
            # Validation
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                
                logger.info(
                    f"Val - Loss: {val_metrics['total_loss']:.4f}, "
                    f"Feature: {val_metrics['feature_loss']:.4f}, "
                    f"Attention: {val_metrics['attention_loss']:.4f}, "
                    f"Intermediate: {val_metrics['intermediate_loss']:.4f}"
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
                    
            print("-" * 70)
    
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

# Enhanced usage example
def main():
    from data.hf_cifar import get_cifar_dataloader
    
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
    
    # Enhanced student model configuration
    student_config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.0,
        'distill_layers': [3, 6, 9],  # Specify which layers to distill
    }
    
    # Initialize enhanced distiller
    distiller = EnhancedDINOv2Distiller(
        student_config=student_config,
        teacher_model_name="facebook/dinov2-small",
        learning_rate=1e-4,
        weight_decay=0.05,
        temperature=4.0,
        attention_temperature=3.0,
        distill_layers=[3, 6, 9]  # Corresponding teacher layers
    )
    
    # Train the model
    distiller.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=50,
        save_path="enhanced_dinov2_student_cifar10.pth"
    )
    
    print("Enhanced training completed!")

if __name__ == "__main__":
    main()