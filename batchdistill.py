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
        # For timm ViT attention modules, we need to capture the attention weights
        # The attention weights are typically stored in module.attention_weights after forward pass
        if hasattr(module, 'attention_weights') and module.attention_weights is not None:
            self.attention_weights.append(module.attention_weights.clone())
        
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
        # Don't register hooks since we'll extract attention manually
        
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights"""
        # We'll implement a custom forward to extract attention weights
        pass
    
    def _extract_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Custom forward pass to extract attention weights"""
        attention_weights = []
        
        # Get patch embeddings
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        
        # Forward through transformer blocks and extract attention
        for i, block in enumerate(self.vit.blocks):
            # Manual attention computation to extract weights
            B, N, C = x.shape
            qkv = block.attn.qkv(x).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attention_weights.append(attn.mean(dim=1))  # Average across heads
            
            # Complete the attention forward pass
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = block.attn.proj(x)
            x = block.attn.proj_drop(x)
            
            # Complete the block forward pass
            x = x + block.drop_path(block.attn(block.norm1(x)))
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        
        return x, attention_weights
    
    def forward(self, x: torch.Tensor, return_features: bool = True) -> Dict[str, torch.Tensor]:
        # Store intermediate features and attention weights
        intermediate_features = {}
        attention_weights = []
        
        # Get patch embeddings
        x_patches = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x_patches.shape[0], -1, -1)
        x_forward = torch.cat((cls_token, x_patches), dim=1)
        x_forward = self.vit.pos_drop(x_forward + self.vit.pos_embed)
        
        # Forward through transformer blocks
        for i, block in enumerate(self.vit.blocks):
            # Extract attention weights manually before block forward
            B, N, C = x_forward.shape
            
            # Store features from specified layers BEFORE applying the block
            if i in self.distill_layers:
                # Extract CLS token and project (before this layer processes it)
                cls_features = x_forward[:, 0]  # [B, embed_dim]
                projected = self.intermediate_projections[str(i)](cls_features)
                intermediate_features[f'layer_{i}'] = {
                    'features': cls_features,
                    'projected_features': projected
                }
            
            # Manually compute attention to extract weights
            try:
                qkv = block.attn.qkv(x_forward).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                attn_weights = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn_weights = attn_weights.softmax(dim=-1)
                
                # Store attention weights (average across heads) - make sure it has gradients
                attention_weights.append(attn_weights.mean(dim=1))
            except Exception as e:
                print(f"Error extracting attention at layer {i}: {e}")
                # Create dummy attention if extraction fails
                attention_weights.append(torch.zeros(B, N, N, device=x_forward.device, requires_grad=True))
            
            # Now do the regular forward pass for this block
            x_forward = block(x_forward)
        
        # Final processing
        x_forward = self.vit.norm(x_forward)
        
        # Extract final CLS token
        cls_token = x_forward[:, 0]  # [B, embed_dim]
        projected_features = self.projection_head(cls_token)
        
        outputs = {
            'features': cls_token,
            'projected_features': projected_features,
            'intermediate_features': intermediate_features,
            'attention_weights': attention_weights
        }
        
        if return_features:
            outputs['patch_features'] = x_forward[:, 1:]  # Exclude CLS token
            
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
        
        # Map student layers to teacher layers (student has 12 layers, teacher might have different)
        if distill_layers is None:
            self.distill_layers = [self.num_layers//4, self.num_layers//2, 3*self.num_layers//4]
        else:
            # Map student layer indices to teacher layer indices
            # If student has 12 layers and teacher has 12, use direct mapping
            # If student has 12 and teacher has 24, map accordingly
            layer_ratio = self.num_layers / 12  # assuming student has 12 layers
            self.distill_layers = [int(layer * layer_ratio) for layer in distill_layers]
            # Ensure we don't exceed teacher layers
            self.distill_layers = [min(layer, self.num_layers - 1) for layer in self.distill_layers]
        
        print(f"Teacher has {self.num_layers} layers, will distill from layers: {self.distill_layers}")
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.dinov2(x, output_hidden_states=True, output_attentions=True)
        
        # Get final CLS token
        cls_token = outputs.last_hidden_state[:, 0]  # [B, embed_dim]
        
        # Extract intermediate features from specified layers
        intermediate_features = {}
        for i, layer_idx in enumerate(self.distill_layers):
            if layer_idx < len(outputs.hidden_states):
                layer_output = outputs.hidden_states[layer_idx]
                cls_features = layer_output[:, 0]  # CLS token from this layer
                # Use the student layer index as key for matching
                student_layer_idx = [3, 6, 9][i] if i < 3 else layer_idx  # Default mapping
                intermediate_features[f'layer_{student_layer_idx}'] = {
                    'features': cls_features,
                }
        
        # Process attention weights (average across heads)
        processed_attentions = []
        if outputs.attentions is not None:
            # Select attention layers corresponding to student layers
            student_layers = [3, 6, 9] if len(self.distill_layers) >= 3 else self.distill_layers
            for layer_idx in student_layers:
                # Map to teacher layer
                teacher_layer_idx = int(layer_idx * len(outputs.attentions) / 12)
                teacher_layer_idx = min(teacher_layer_idx, len(outputs.attentions) - 1)
                
                if teacher_layer_idx < len(outputs.attentions):
                    attn = outputs.attentions[teacher_layer_idx]
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
        attention_loss_weight: float = 0.1,  # Reduced since MSE produces larger values
        intermediate_loss_weight: float = 0.01,  # Reduced since we're scaling by 100x
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
        """Attention transfer loss using MSE (more aggressive than KL divergence)"""
        # Get device from one of the tensors or use CPU as fallback
        device = 'cpu'
        if student_attentions and len(student_attentions) > 0:
            device = student_attentions[0].device
        elif teacher_attentions and len(teacher_attentions) > 0:
            device = teacher_attentions[0].device
            
        if not student_attentions or not teacher_attentions:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        if len(student_attentions) == 0 or len(teacher_attentions) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_valid_layers = 0
        
        # Map student layers to teacher layers
        student_layer_indices = [3, 6, 9]  # Which student layers to use
        
        for i, student_layer_idx in enumerate(student_layer_indices):
            if i >= len(teacher_attentions) or student_layer_idx >= len(student_attentions):
                continue
                
            try:
                student_attn = student_attentions[student_layer_idx]
                teacher_attn = teacher_attentions[i]
                
                # Skip if tensors are None or empty
                if student_attn is None or teacher_attn is None:
                    continue
                    
                if student_attn.numel() == 0 or teacher_attn.numel() == 0:
                    continue
                
                # Ensure both are 3D [B, seq, seq] and contiguous
                if len(student_attn.shape) == 4:  # [B, heads, seq, seq]
                    student_attn = student_attn.mean(dim=1)
                if len(teacher_attn.shape) == 4:  # [B, heads, seq, seq]
                    teacher_attn = teacher_attn.mean(dim=1)
                
                # Make tensors contiguous
                student_attn = student_attn.contiguous()
                teacher_attn = teacher_attn.contiguous()
                
                # Handle different sequence lengths
                if student_attn.shape[-1] != teacher_attn.shape[-1]:
                    min_seq = min(student_attn.shape[-1], teacher_attn.shape[-1])
                    student_attn = student_attn[:, :min_seq, :min_seq]
                    teacher_attn = teacher_attn[:, :min_seq, :min_seq]
                
                if student_attn.shape[-1] <= 1:
                    continue
                
                # NEW APPROACH: Use MSE loss directly on attention weights
                # This should produce much larger gradients than KL divergence
                layer_loss = F.mse_loss(student_attn, teacher_attn)
                
                # Scale by attention dimension to make loss magnitude reasonable
                scale_factor = student_attn.shape[-1]  # 197
                layer_loss = layer_loss * scale_factor
                
                if not torch.isnan(layer_loss) and not torch.isinf(layer_loss):
                    total_loss = total_loss + layer_loss
                    num_valid_layers += 1
                
            except Exception as e:
                continue
                
        final_loss = total_loss / max(num_valid_layers, 1) if num_valid_layers > 0 else torch.tensor(0.0, device=device, requires_grad=True)
        return final_loss
    
    def intermediate_layer_loss(
        self,
        student_intermediate: Dict[str, Dict[str, torch.Tensor]],
        teacher_intermediate: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Loss for intermediate layer features - using unnormalized MSE"""
        # Get device from first available tensor
        device = 'cpu'
        for layer_key in student_intermediate:
            if 'projected_features' in student_intermediate[layer_key]:
                device = student_intermediate[layer_key]['projected_features'].device
                break
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_layers = 0
        
        # Return early if no features
        if len(student_intermediate) == 0 or len(teacher_intermediate) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        for layer_key in student_intermediate:
            if layer_key in teacher_intermediate:
                try:
                    student_feat = student_intermediate[layer_key]['projected_features']
                    teacher_feat = teacher_intermediate[layer_key]['features']
                    
                    # Check tensor validity
                    if student_feat is None or teacher_feat is None:
                        continue
                    if student_feat.numel() == 0 or teacher_feat.numel() == 0:
                        continue
                    
                    # Handle dimension mismatch
                    if student_feat.shape[-1] != teacher_feat.shape[-1]:
                        # Project teacher features to student dimension
                        if not hasattr(self, f'teacher_proj_{layer_key}'):
                            proj_layer = nn.Linear(teacher_feat.shape[-1], student_feat.shape[-1]).to(student_feat.device)
                            setattr(self, f'teacher_proj_{layer_key}', proj_layer)
                        proj_layer = getattr(self, f'teacher_proj_{layer_key}')
                        teacher_feat = proj_layer(teacher_feat)
                    
                    # NEW APPROACH: Use raw MSE without normalization
                    # Normalization was constraining the loss to be very small
                    layer_loss = F.mse_loss(student_feat, teacher_feat)
                    
                    # Scale the loss to make it more significant
                    layer_loss = layer_loss * 100.0  # Scale by 100x
                    
                    if not torch.isnan(layer_loss) and not torch.isinf(layer_loss):
                        total_loss = total_loss + layer_loss
                        num_layers += 1
                        
                except Exception as e:
                    continue
        
        final_loss = total_loss / max(num_layers, 1) if num_layers > 0 else torch.tensor(0.0, device=device, requires_grad=True)
        return final_loss
    
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
        
        def safe_item(tensor_or_float):
            """Safely convert tensor or float to float"""
            if isinstance(tensor_or_float, torch.Tensor):
                return tensor_or_float.item()
            return float(tensor_or_float)
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                teacher_outputs = self.teacher(images)
                
            student_outputs = self.student(images)
            
            # Debug info only for first batch
            if batch_idx == 0:
                print(f"\n=== ENHANCED LOSS DEBUG INFO ===")
                print(f"Student attention weights: {len(student_outputs.get('attention_weights', []))}")
                print(f"Teacher attention weights: {len(teacher_outputs.get('attention_weights', []))}")
                print(f"Student intermediate features: {list(student_outputs.get('intermediate_features', {}).keys())}")
                print(f"Teacher intermediate features: {list(teacher_outputs.get('intermediate_features', {}).keys())}")
                
                # Check attention weights
                if student_outputs.get('attention_weights'):
                    s_attn = student_outputs['attention_weights'][3]  # Check layer 3
                    print(f"Student attention (layer 3) shape: {s_attn.shape}")
                    print(f"Student attention mean: {s_attn.mean().item():.6f}, std: {s_attn.std().item():.6f}")
                
                if teacher_outputs.get('attention_weights'):
                    t_attn = teacher_outputs['attention_weights'][0]  # Check first teacher layer
                    print(f"Teacher attention (layer 0) shape: {t_attn.shape}")
                    print(f"Teacher attention mean: {t_attn.mean().item():.6f}, std: {t_attn.std().item():.6f}")
                print("================================\n")
            
            # Compute losses
            losses = self.criterion(student_outputs, teacher_outputs)
            
            # Debug individual losses for first batch
            if batch_idx == 0:
                print(f"ENHANCED Loss breakdown:")
                print(f"  Feature loss: {safe_item(losses['feature_loss']):.6f}")
                print(f"  Cosine loss: {safe_item(losses['cosine_loss']):.6f}")
                print(f"  Attention loss: {safe_item(losses['attention_loss']):.6f}")
                print(f"  Intermediate loss: {safe_item(losses['intermediate_loss']):.6f}")
                print(f"  Total loss: {safe_item(losses['total_loss']):.6f}\n")
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics with safe conversion
            total_loss += safe_item(losses['total_loss'])
            total_feature_loss += safe_item(losses['feature_loss'])
            total_cosine_loss += safe_item(losses['cosine_loss'])
            total_attention_loss += safe_item(losses['attention_loss'])
            total_intermediate_loss += safe_item(losses['intermediate_loss'])
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{safe_item(losses['total_loss']):.4f}",
                'Feat': f"{safe_item(losses['feature_loss']):.4f}",
                'Attn': f"{safe_item(losses['attention_loss']):.4f}",
                'Inter': f"{safe_item(losses['intermediate_loss']):.4f}",
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
        
        def safe_item(tensor_or_float):
            """Safely convert tensor or float to float"""
            if isinstance(tensor_or_float, torch.Tensor):
                return tensor_or_float.item()
            return float(tensor_or_float)
        
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                
                teacher_outputs = self.teacher(images)
                student_outputs = self.student(images)
                
                losses = self.criterion(student_outputs, teacher_outputs)
                
                total_loss += safe_item(losses['total_loss'])
                total_feature_loss += safe_item(losses['feature_loss'])
                total_cosine_loss += safe_item(losses['cosine_loss'])
                total_attention_loss += safe_item(losses['attention_loss'])
                total_intermediate_loss += safe_item(losses['intermediate_loss'])
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