import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveEvaluator:
    """Complete evaluation and visualization suite for student vs teacher comparison"""
    
    def __init__(
        self,
        student_model_path: str,
        student_config: Dict,
        teacher_model_name: str = "facebook/dinov2-small",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "evaluation_results01"
    ):
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"🔧 Initializing models on {device}...")
        
        # Import your models here - adjust paths as needed
        try:
            from batchdistill import StudentViT, DINOv2Teacher
        except ImportError:
            print("⚠️  Please ensure your model classes are importable")
            print("   You may need to adjust the import statement above")
            raise
        
        # Load teacher model
        self.teacher = DINOv2Teacher(teacher_model_name).to(device)
        self.teacher.eval()
        
        # Load student model
        self.student = StudentViT(**student_config).to(device)
        
        # Load trained weights
        if os.path.exists(student_model_path):
            checkpoint = torch.load(student_model_path, map_location=device)
            self.student.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Student model loaded from {student_model_path}")
        else:
            print(f"⚠️  Student model not found at {student_model_path}")
            
        self.student.eval()
        
        # Storage for evaluation results
        self.results = {
            'student_features': [],
            'teacher_features': [],
            'student_intermediate': [],
            'teacher_intermediate': [],
            'student_attentions': [],
            'teacher_attentions': [],
            'images': [],
            'labels': [],
            'predictions_student': [],
            'predictions_teacher': [],
            'similarities': []
        }
        
        print("✅ Evaluator initialized successfully!")
    
    def extract_comprehensive_features(self, dataloader, max_batches: int = 20):
        """Extract detailed features from both models"""
        print(f"🔍 Extracting features from {max_batches} batches...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Processing")):
                if batch_idx >= max_batches:
                    break
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get teacher outputs
                teacher_out = self.teacher(images)
                
                # Get student outputs
                student_out = self.student(images)
                
                # Compute similarities for each sample
                student_feat = student_out['features'].cpu().numpy()
                teacher_feat = teacher_out['features'].cpu().numpy()
                
                batch_similarities = []
                for i in range(len(student_feat)):
                    sim = cosine_similarity([student_feat[i]], [teacher_feat[i]])[0][0]
                    batch_similarities.append(sim)
                
                # Store all results
                self.results['student_features'].append(student_out['features'].cpu())
                self.results['teacher_features'].append(teacher_out['features'].cpu())
                self.results['images'].append(images.cpu())
                self.results['labels'].append(labels.cpu())
                self.results['similarities'].extend(batch_similarities)
                
                # Store intermediate features if available
                if 'intermediate_features' in student_out:
                    self.results['student_intermediate'].append(student_out['intermediate_features'])
                if 'intermediate_features' in teacher_out:
                    self.results['teacher_intermediate'].append(teacher_out['intermediate_features'])
                
                # Store attention weights if available
                if 'attention_weights' in student_out and student_out['attention_weights']:
                    self.results['student_attentions'].extend([attn.cpu() for attn in student_out['attention_weights']])
                if 'attention_weights' in teacher_out and teacher_out['attention_weights']:
                    self.results['teacher_attentions'].extend([attn.cpu() for attn in teacher_out['attention_weights']])
        
        # Concatenate features
        self.results['student_features'] = torch.cat(self.results['student_features'], dim=0)
        self.results['teacher_features'] = torch.cat(self.results['teacher_features'], dim=0)
        self.results['images'] = torch.cat(self.results['images'], dim=0)
        self.results['labels'] = torch.cat(self.results['labels'], dim=0)
        
        print(f"✅ Extracted features for {len(self.results['student_features'])} samples")
        return self.results
    
    def plot_similarity_analysis(self):
        """Comprehensive similarity analysis with multiple views"""
        similarities = np.array(self.results['similarities'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Student-Teacher Feature Similarity Analysis', fontsize=16, fontweight='bold')
        
        # 1. Distribution histogram
        axes[0, 0].hist(similarities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(similarities):.3f}')
        axes[0, 0].axvline(np.median(similarities), color='orange', linestyle='--', linewidth=2,
                          label=f'Median: {np.median(similarities):.3f}')
        axes[0, 0].set_title('Similarity Distribution')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot by label
        labels = self.results['labels'].numpy()
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 20:  # Only if manageable number of classes
            similarity_by_label = [similarities[labels == label] for label in unique_labels]
            axes[0, 1].boxplot(similarity_by_label, labels=unique_labels)
            axes[0, 1].set_title('Similarity by Class')
            axes[0, 1].set_xlabel('Class Label')
            axes[0, 1].set_ylabel('Cosine Similarity')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            # Scatter plot if too many classes
            axes[0, 1].scatter(range(len(similarities)), similarities, alpha=0.6, s=10)
            axes[0, 1].set_title('Similarity per Sample')
            axes[0, 1].set_xlabel('Sample Index')
            axes[0, 1].set_ylabel('Cosine Similarity')
        
        # 3. Cumulative distribution
        sorted_sims = np.sort(similarities)
        axes[0, 2].plot(sorted_sims, np.linspace(0, 1, len(sorted_sims)), linewidth=2)
        axes[0, 2].set_title('Cumulative Distribution')
        axes[0, 2].set_xlabel('Cosine Similarity')
        axes[0, 2].set_ylabel('Cumulative Probability')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Feature magnitude comparison
        student_norms = torch.norm(self.results['student_features'], dim=1).numpy()
        teacher_norms = torch.norm(self.results['teacher_features'], dim=1).numpy()
        axes[1, 0].scatter(teacher_norms, student_norms, alpha=0.6, s=20)
        axes[1, 0].plot([teacher_norms.min(), teacher_norms.max()], 
                       [teacher_norms.min(), teacher_norms.max()], 'r--', alpha=0.8)
        axes[1, 0].set_title('Feature Magnitude Comparison')
        axes[1, 0].set_xlabel('Teacher Feature Norm')
        axes[1, 0].set_ylabel('Student Feature Norm')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Similarity vs Feature Distance
        feature_distances = torch.norm(self.results['student_features'] - self.results['teacher_features'], dim=1).numpy()
        axes[1, 1].scatter(feature_distances, similarities, alpha=0.6, s=20)
        axes[1, 1].set_title('Similarity vs Feature Distance')
        axes[1, 1].set_xlabel('L2 Feature Distance')
        axes[1, 1].set_ylabel('Cosine Similarity')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Quality assessment
        quality_ranges = {
            'Excellent (>0.9)': np.sum(similarities > 0.9),
            'Very Good (0.8-0.9)': np.sum((similarities > 0.8) & (similarities <= 0.9)),
            'Good (0.7-0.8)': np.sum((similarities > 0.7) & (similarities <= 0.8)),
            'Moderate (0.6-0.7)': np.sum((similarities > 0.6) & (similarities <= 0.7)),
            'Poor (<0.6)': np.sum(similarities <= 0.6)
        }
        
        colors = ['darkgreen', 'green', 'orange', 'red', 'darkred']
        axes[1, 2].pie(quality_ranges.values(), labels=quality_ranges.keys(), autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[1, 2].set_title('Quality Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'similarity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"📊 SIMILARITY STATISTICS:")
        print(f"   Mean: {np.mean(similarities):.3f} ± {np.std(similarities):.3f}")
        print(f"   Median: {np.median(similarities):.3f}")
        print(f"   Min: {np.min(similarities):.3f}")
        print(f"   Max: {np.max(similarities):.3f}")
        print(f"   Samples > 0.8: {np.sum(similarities > 0.8)}/{len(similarities)} ({100*np.sum(similarities > 0.8)/len(similarities):.1f}%)")
    
    def plot_feature_space_comparison(self):
        """Advanced feature space visualization"""
        print("🎨 Creating feature space visualizations...")
        
        student_feat = self.results['student_features'].numpy()
        teacher_feat = self.results['teacher_features'].numpy()
        labels = self.results['labels'].numpy()
        
        # Subsample for visualization if too many points
        n_samples = min(2000, len(student_feat))
        indices = np.random.choice(len(student_feat), n_samples, replace=False)
        
        student_subset = student_feat[indices]
        teacher_subset = teacher_feat[indices]
        labels_subset = labels[indices]
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Feature Space Comparison: Student vs Teacher', fontsize=16, fontweight='bold')
        
        # PCA Analysis
        print("   Computing PCA...")
        pca_student = PCA(n_components=2)
        pca_teacher = PCA(n_components=2)
        
        student_pca = pca_student.fit_transform(student_subset)
        teacher_pca = pca_teacher.fit_transform(teacher_subset)
        
        # Plot PCA
        unique_labels = np.unique(labels_subset)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels_subset == label
            axes[0, 0].scatter(student_pca[mask, 0], student_pca[mask, 1], 
                             c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
        axes[0, 0].set_title(f'Student Features (PCA)\nExplained Variance: {pca_student.explained_variance_ratio_.sum():.2f}')
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        if len(unique_labels) <= 10:
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for i, label in enumerate(unique_labels):
            mask = labels_subset == label
            axes[0, 1].scatter(teacher_pca[mask, 0], teacher_pca[mask, 1], 
                             c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
        axes[0, 1].set_title(f'Teacher Features (PCA)\nExplained Variance: {pca_teacher.explained_variance_ratio_.sum():.2f}')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        
        # t-SNE Analysis
        print("   Computing t-SNE...")
        tsne_student = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//4))
        tsne_teacher = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//4))
        
        student_tsne = tsne_student.fit_transform(student_subset)
        teacher_tsne = tsne_teacher.fit_transform(teacher_subset)
        
        for i, label in enumerate(unique_labels):
            mask = labels_subset == label
            axes[1, 0].scatter(student_tsne[mask, 0], student_tsne[mask, 1], 
                             c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
        axes[1, 0].set_title('Student Features (t-SNE)')
        axes[1, 0].set_xlabel('t-SNE 1')
        axes[1, 0].set_ylabel('t-SNE 2')
        
        for i, label in enumerate(unique_labels):
            mask = labels_subset == label
            axes[1, 1].scatter(teacher_tsne[mask, 0], teacher_tsne[mask, 1], 
                             c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
        axes[1, 1].set_title('Teacher Features (t-SNE)')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        
        # Cluster analysis
        print("   Computing cluster analysis...")
        n_clusters = min(len(unique_labels), 10)
        
        kmeans_student = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_teacher = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        student_clusters = kmeans_student.fit_predict(student_subset)
        teacher_clusters = kmeans_teacher.fit_predict(teacher_subset)
        
        # Plot clusters on PCA space
        scatter = axes[0, 2].scatter(student_pca[:, 0], student_pca[:, 1], 
                                   c=student_clusters, cmap='viridis', alpha=0.6, s=20)
        axes[0, 2].set_title(f'Student Clusters (K={n_clusters})')
        axes[0, 2].set_xlabel('PC1')
        axes[0, 2].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[0, 2])
        
        scatter = axes[1, 2].scatter(teacher_pca[:, 0], teacher_pca[:, 1], 
                                   c=teacher_clusters, cmap='viridis', alpha=0.6, s=20)
        axes[1, 2].set_title(f'Teacher Clusters (K={n_clusters})')
        axes[1, 2].set_xlabel('PC1')
        axes[1, 2].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'feature_space_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_sample_comparisons(self, num_samples: int = 12):
        """Show side-by-side comparisons of how student and teacher process samples"""
        print(f"🖼️  Creating sample comparison visualizations...")
        
        # Select diverse samples (high, medium, low similarity)
        similarities = np.array(self.results['similarities'])
        
        # Get samples from different similarity ranges
        high_sim_idx = np.where(similarities > np.percentile(similarities, 80))[0]
        med_sim_idx = np.where((similarities > np.percentile(similarities, 40)) & 
                              (similarities < np.percentile(similarities, 60)))[0]
        low_sim_idx = np.where(similarities < np.percentile(similarities, 20))[0]
        
        # Select samples
        selected_indices = []
        for indices, count in [(high_sim_idx, 4), (med_sim_idx, 4), (low_sim_idx, 4)]:
            if len(indices) >= count:
                selected_indices.extend(np.random.choice(indices, count, replace=False))
            else:
                selected_indices.extend(indices)
        
        # Ensure we have enough samples
        while len(selected_indices) < num_samples and len(selected_indices) < len(similarities):
            remaining = list(set(range(len(similarities))) - set(selected_indices))
            selected_indices.extend(np.random.choice(remaining, 
                                                   min(num_samples - len(selected_indices), len(remaining)), 
                                                   replace=False))
        
        selected_indices = selected_indices[:num_samples]
        
        # Create the visualization
        n_cols = 4  # Image, Student attention, Teacher attention, Similarity bar
        n_rows = len(selected_indices)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Sample-by-Sample Comparison: Student vs Teacher Processing', 
                    fontsize=16, fontweight='bold')
        
        for i, idx in enumerate(selected_indices):
            # Original image
            img = self.results['images'][idx]
            img_denorm = self.denormalize_image(img)
            axes[i, 0].imshow(img_denorm.permute(1, 2, 0).numpy())
            axes[i, 0].set_title(f'Sample {idx}\nLabel: {self.results["labels"][idx].item()}\n'
                               f'Similarity: {similarities[idx]:.3f}')
            axes[i, 0].axis('off')
            
            # Student attention (if available)
            if (self.results['student_attentions'] and 
                idx < len(self.results['student_attentions'])):
                try:
                    student_attn = self.results['student_attentions'][idx]
                    if len(student_attn.shape) >= 2:
                        # Use CLS token attention to patches
                        if len(student_attn.shape) == 3:
                            cls_attn = student_attn[0, 1:]  # CLS to patches
                        else:
                            cls_attn = student_attn[1:]  # Skip CLS
                        
                        attn_map = self.attention_to_image(cls_attn, img_size=224)
                        
                        axes[i, 1].imshow(img_denorm.permute(1, 2, 0).numpy())
                        im = axes[i, 1].imshow(attn_map, alpha=0.6, cmap='hot')
                        axes[i, 1].set_title('Student Attention')
                        axes[i, 1].axis('off')
                    else:
                        axes[i, 1].text(0.5, 0.5, 'No attention\navailable', 
                                       ha='center', va='center', transform=axes[i, 1].transAxes)
                        axes[i, 1].axis('off')
                except Exception as e:
                    axes[i, 1].text(0.5, 0.5, f'Attention\nprocessing\nerror', 
                                   ha='center', va='center', transform=axes[i, 1].transAxes)
                    axes[i, 1].axis('off')
            else:
                axes[i, 1].text(0.5, 0.5, 'No student\nattention', 
                               ha='center', va='center', transform=axes[i, 1].transAxes)
                axes[i, 1].axis('off')
            
            # Teacher attention (if available)
            if (self.results['teacher_attentions'] and 
                idx < len(self.results['teacher_attentions'])):
                try:
                    teacher_attn = self.results['teacher_attentions'][idx]
                    if len(teacher_attn.shape) >= 2:
                        # Use CLS token attention to patches
                        if len(teacher_attn.shape) == 3:
                            cls_attn = teacher_attn[0, 1:]  # CLS to patches
                        else:
                            cls_attn = teacher_attn[1:]  # Skip CLS
                        
                        attn_map = self.attention_to_image(cls_attn, img_size=224)
                        
                        axes[i, 2].imshow(img_denorm.permute(1, 2, 0).numpy())
                        axes[i, 2].imshow(attn_map, alpha=0.6, cmap='hot')
                        axes[i, 2].set_title('Teacher Attention')
                        axes[i, 2].axis('off')
                    else:
                        axes[i, 2].text(0.5, 0.5, 'No attention\navailable', 
                                       ha='center', va='center', transform=axes[i, 2].transAxes)
                        axes[i, 2].axis('off')
                except Exception as e:
                    axes[i, 2].text(0.5, 0.5, f'Attention\nprocessing\nerror', 
                                   ha='center', va='center', transform=axes[i, 2].transAxes)
                    axes[i, 2].axis('off')
            else:
                axes[i, 2].text(0.5, 0.5, 'No teacher\nattention', 
                               ha='center', va='center', transform=axes[i, 2].transAxes)
                axes[i, 2].axis('off')
            
            # Similarity bar
            sim_val = similarities[idx]
            color = plt.cm.RdYlGn(sim_val)  # Red to Green based on similarity
            axes[i, 3].barh([0], [sim_val], color=color, alpha=0.7)
            axes[i, 3].set_xlim(0, 1)
            axes[i, 3].set_ylim(-0.5, 0.5)
            axes[i, 3].set_xlabel('Similarity')
            axes[i, 3].set_title(f'Sim: {sim_val:.3f}')
            axes[i, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'sample_comparisons.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_attention_analysis(self):
        """Detailed attention pattern analysis"""
        if not self.results['student_attentions'] or not self.results['teacher_attentions']:
            print("⚠️  No attention data available for analysis")
            return
        
        print("🎯 Analyzing attention patterns...")
        
        # Compute attention similarities
        attention_similarities = []
        min_len = min(len(self.results['student_attentions']), 
                     len(self.results['teacher_attentions']))
        
        for i in range(min_len):
            try:
                s_attn = self.results['student_attentions'][i]
                t_attn = self.results['teacher_attentions'][i]
                
                if s_attn.shape == t_attn.shape and s_attn.numel() > 0:
                    sim = F.cosine_similarity(s_attn.flatten().unsqueeze(0), 
                                            t_attn.flatten().unsqueeze(0)).item()
                    attention_similarities.append(sim)
            except Exception:
                continue
        
        if not attention_similarities:
            print("⚠️  Could not compute attention similarities")
            return
        
        # Create attention analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Attention Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Attention similarity distribution
        axes[0, 0].hist(attention_similarities, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 0].axvline(np.mean(attention_similarities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(attention_similarities):.3f}')
        axes[0, 0].set_title('Attention Pattern Similarity')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Feature vs Attention similarity correlation
        feature_sims = self.results['similarities'][:len(attention_similarities)]
        axes[0, 1].scatter(feature_sims, attention_similarities, alpha=0.6, s=20)
        axes[0, 1].set_xlabel('Feature Similarity')
        axes[0, 1].set_ylabel('Attention Similarity')
        axes[0, 1].set_title('Feature vs Attention Similarity')
        
        # Add correlation coefficient
        corr = np.corrcoef(feature_sims, attention_similarities)[0, 1]
        axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[0, 1].transAxes, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average attention maps
        if len(self.results['student_attentions']) > 0:
            try:
                # Compute average attention patterns
                student_avg_attn = torch.stack(self.results['student_attentions'][:10]).mean(0)
                teacher_avg_attn = torch.stack(self.results['teacher_attentions'][:10]).mean(0)
                
                # Plot average student attention
                if len(student_avg_attn.shape) >= 2:
                    attn_to_plot = student_avg_attn[0] if len(student_avg_attn.shape) == 3 else student_avg_attn
                    im1 = axes[1, 0].imshow(attn_to_plot.numpy(), cmap='hot')
                    axes[1, 0].set_title('Average Student Attention')
                    plt.colorbar(im1, ax=axes[1, 0])
                
                # Plot average teacher attention  
                if len(teacher_avg_attn.shape) >= 2:
                    attn_to_plot = teacher_avg_attn[0] if len(teacher_avg_attn.shape) == 3 else teacher_avg_attn
                    im2 = axes[1, 1].imshow(attn_to_plot.numpy(), cmap='hot')
                    axes[1, 1].set_title('Average Teacher Attention')
                    plt.colorbar(im2, ax=axes[1, 1])
                    
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, 'Could not compute\naverage attention', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 1].text(0.5, 0.5, 'Could not compute\naverage attention', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'attention_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 ATTENTION STATISTICS:")
        print(f"   Mean attention similarity: {np.mean(attention_similarities):.3f} ± {np.std(attention_similarities):.3f}")
        print(f"   Correlation with features: {corr:.3f}")
    
    def attention_to_image(self, attention_weights, img_size=224, patch_size=16):
        """Convert attention weights to spatial attention map"""
        try:
            # Calculate grid size
            grid_size = img_size // patch_size
            
            # Handle different attention shapes
            if len(attention_weights.shape) == 1:
                # Flatten attention vector
                attn_flat = attention_weights.numpy()
            else:
                attn_flat = attention_weights.flatten().numpy()
            
            # Expected number of patches
            expected_patches = grid_size * grid_size
            
            # Take only the number of patches we need
            if len(attn_flat) >= expected_patches:
                attn_flat = attn_flat[:expected_patches]
            else:
                # Pad if needed
                attn_flat = np.pad(attn_flat, (0, expected_patches - len(attn_flat)), 'constant')
            
            # Reshape to spatial grid
            attention_map = attn_flat.reshape(grid_size, grid_size)
            
            # Resize to image size
            attention_map = cv2.resize(attention_map, (img_size, img_size))
            
            return attention_map
        except Exception as e:
            # Return empty map if processing fails
            return np.zeros((img_size, img_size))
    
    def denormalize_image(self, tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Denormalize image tensor for visualization"""
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("📝 Generating comprehensive evaluation report...")
        
        similarities = np.array(self.results['similarities'])
        
        # Performance metrics
        metrics = {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'median_similarity': np.median(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'samples_total': len(similarities),
            'samples_excellent': np.sum(similarities > 0.9),
            'samples_very_good': np.sum((similarities > 0.8) & (similarities <= 0.9)),
            'samples_good': np.sum((similarities > 0.7) & (similarities <= 0.8)),
            'samples_moderate': np.sum((similarities > 0.6) & (similarities <= 0.7)),
            'samples_poor': np.sum(similarities <= 0.6)
        }
        
        # Create detailed report
        report = f"""
# DINOv2 Distillation Evaluation Report

## Summary
This report analyzes the performance of a student model distilled from DINOv2.

## Overall Performance
- **Mean Similarity**: {metrics['mean_similarity']:.3f} ± {metrics['std_similarity']:.3f}
- **Median Similarity**: {metrics['median_similarity']:.3f}
- **Range**: [{metrics['min_similarity']:.3f}, {metrics['max_similarity']:.3f}]

## Quality Distribution
- **Excellent (>0.9)**: {metrics['samples_excellent']}/{metrics['samples_total']} ({100*metrics['samples_excellent']/metrics['samples_total']:.1f}%)
- **Very Good (0.8-0.9)**: {metrics['samples_very_good']}/{metrics['samples_total']} ({100*metrics['samples_very_good']/metrics['samples_total']:.1f}%)
- **Good (0.7-0.8)**: {metrics['samples_good']}/{metrics['samples_total']} ({100*metrics['samples_good']/metrics['samples_total']:.1f}%)
- **Moderate (0.6-0.7)**: {metrics['samples_moderate']}/{metrics['samples_total']} ({100*metrics['samples_moderate']/metrics['samples_total']:.1f}%)
- **Poor (<0.6)**: {metrics['samples_poor']}/{metrics['samples_total']} ({100*metrics['samples_poor']/metrics['samples_total']:.1f}%)

## Model Information
- **Student Features**: {self.results['student_features'].shape[1]} dimensions
- **Teacher Features**: {self.results['teacher_features'].shape[1]} dimensions
- **Samples Analyzed**: {metrics['samples_total']}
- **Device**: {self.device}

## Performance Assessment
"""
        
        # Add performance assessment
        if metrics['mean_similarity'] > 0.85:
            report += "🟢 **EXCELLENT**: The student model has learned very well from the teacher.\n"
        elif metrics['mean_similarity'] > 0.75:
            report += "🟡 **GOOD**: The student model shows good alignment with the teacher.\n"
        elif metrics['mean_similarity'] > 0.65:
            report += "🟠 **MODERATE**: The student model has partial alignment. Consider tuning hyperparameters.\n"
        else:
            report += "🔴 **POOR**: The student model needs significant improvement. Review training process.\n"
        
        report += f"""
## Recommendations
- For similarity > 0.8: Consider the distillation successful
- For similarity 0.6-0.8: Try increasing training epochs or adjusting loss weights
- For similarity < 0.6: Review model architecture, learning rate, or distillation strategy

## Files Generated
- `similarity_analysis.png`: Comprehensive similarity analysis
- `feature_space_comparison.png`: PCA and t-SNE visualizations  
- `sample_comparisons.png`: Side-by-side sample processing comparisons
- `attention_analysis.png`: Attention pattern analysis
- `evaluation_report.md`: This report

## Training Recommendations
Based on the results, consider:
1. **High similarity (>0.8)**: Distillation successful, try reducing model size further
2. **Medium similarity (0.6-0.8)**: Increase attention/intermediate loss weights
3. **Low similarity (<0.6)**: Review architecture compatibility or increase training time

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        with open(os.path.join(self.save_dir, 'evaluation_report.md'), 'w') as f:
            f.write(report)
        
        print(f"✅ Evaluation complete! Results saved to '{self.save_dir}/'")
        print(f"📊 Mean similarity: {metrics['mean_similarity']:.3f}")
        print(f"🎯 {metrics['samples_excellent'] + metrics['samples_very_good']} out of {metrics['samples_total']} samples have similarity > 0.8")
        
        return metrics, report

def main():
    """Main evaluation script"""
    print("🚀 Starting DINOv2 Distillation Evaluation")
    print("="*50)
    
    # Configuration - adjust these paths and parameters
    student_model_path = "enhanced_dinov2_student_food101.pth"  # Path to your trained model
    save_directory = "distillation_evaluation_results"
    
    # Student model configuration (should match training config)
    student_config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.0,
        'distill_layers': [3, 6, 9],
    }
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        student_model_path=student_model_path,
        student_config=student_config,
        teacher_model_name="facebook/dinov2-small",
        save_dir=save_directory
    )
    
    # Load test dataset - choose one of the better datasets
    print("\n📚 Loading test dataset...")
    
    # Option 1: Food101 (recommended for high quality)
    from data.multiple_datasets import get_food101_dataloader
    test_loader = get_food101_dataloader(split="validation[:10%]", batch_size=16, shuffle=False)
    
    # Option 2: STL10 (good balance)
    # from improved_datasets import get_stl10_dataloader  
    # test_loader = get_stl10_dataloader(split="test", batch_size=16, shuffle=False)
    
    # Option 3: Oxford Pets
    # from improved_datasets import get_oxford_pets_dataloader
    # test_loader = get_oxford_pets_dataloader(split="test", batch_size=16, shuffle=False)
    
    # Extract features and run evaluation
    print("\n🔍 Extracting features for evaluation...")
    evaluator.extract_comprehensive_features(test_loader, max_batches=15)
    
    print("\n📊 Creating visualizations...")
    
    # Generate all analysis plots
    evaluator.plot_similarity_analysis()
    evaluator.plot_feature_space_comparison()
    evaluator.visualize_sample_comparisons(num_samples=8)
    evaluator.plot_attention_analysis()
    
    # Generate final report
    print("\n📝 Generating final report...")
    metrics, report = evaluator.generate_evaluation_report()
    
    print("\n" + "="*50)
    print("🎉 EVALUATION COMPLETE!")
    print(f"📁 Results saved to: {save_directory}/")
    print(f"📈 Overall Performance: {metrics['mean_similarity']:.3f}")
    
    if metrics['mean_similarity'] > 0.8:
        print("🎯 Excellent distillation! Your student learned very well.")
    elif metrics['mean_similarity'] > 0.7:
        print("👍 Good distillation! Consider fine-tuning for better results.")
    else:
        print("⚠️  Consider reviewing training parameters for improvement.")
    
    print("="*50)

if __name__ == "__main__":
    main()