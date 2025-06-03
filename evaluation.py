import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, fall back gracefully if not available
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        print("Warning: UMAP not available. Install with: pip install umap-learn")
        print("UMAP visualizations will be skipped.")

# Import your models (assuming they're in the same directory or installed)
from batchdistill import StudentViT, DINOv2Teacher, EnhancedDINOv2Distiller

class DistillationEvaluator:
    """Comprehensive evaluation and visualization for distilled models"""
    
    def __init__(
        self,
        student_model_path: str,
        student_config: Dict,
        teacher_model_name: str = "facebook/dinov2-small",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "evaluation_results"
    ):
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Load models
        self.student = StudentViT(**student_config).to(device)
        self.teacher = DINOv2Teacher(teacher_model_name).to(device)
        
        # Load trained student weights
        checkpoint = torch.load(student_model_path, map_location=device)
        self.student.load_state_dict(checkpoint['model_state_dict'])
        
        self.student.eval()
        self.teacher.eval()
        
        print(f"Models loaded successfully!")
        print(f"Results will be saved to: {save_dir}")
    
    def extract_features(
        self, 
        dataloader, 
        max_samples: int = 1000
    ) -> Tuple[Dict[str, np.ndarray], List[int]]:
        """Extract features from both models"""
        
        student_features = []
        teacher_features = []
        student_projected = []
        labels = []
        
        samples_collected = 0
        
        with torch.no_grad():
            for images, batch_labels in tqdm(dataloader, desc="Extracting features"):
                if samples_collected >= max_samples:
                    break
                    
                images = images.to(self.device)
                
                # Get features from both models
                teacher_outputs = self.teacher(images)
                student_outputs = self.student(images)
                
                # Store features
                student_features.append(student_outputs['features'].cpu().numpy())
                teacher_features.append(teacher_outputs['features'].cpu().numpy())
                student_projected.append(student_outputs['projected_features'].cpu().numpy())
                labels.extend(batch_labels.tolist())
                
                samples_collected += len(images)
        
        # Concatenate all features
        features = {
            'student': np.vstack(student_features)[:max_samples],
            'teacher': np.vstack(teacher_features)[:max_samples],
            'student_projected': np.vstack(student_projected)[:max_samples]
        }
        
        labels = labels[:max_samples]
        
        print(f"Extracted features for {len(labels)} samples")
        return features, labels
    
    def compute_similarity_metrics(
        self, 
        features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Compute various similarity metrics between student and teacher"""
        
        student_feats = features['student']
        teacher_feats = features['teacher']
        student_proj = features['student_projected']
        
        # Normalize features
        student_norm = F.normalize(torch.from_numpy(student_feats), dim=1).numpy()
        teacher_norm = F.normalize(torch.from_numpy(teacher_feats), dim=1).numpy()
        student_proj_norm = F.normalize(torch.from_numpy(student_proj), dim=1).numpy()
        
        metrics = {}
        
        # Cosine similarity
        cos_sim_raw = np.mean([
            cosine_similarity([s], [t])[0, 0] 
            for s, t in zip(student_norm, teacher_norm)
        ])
        
        cos_sim_proj = np.mean([
            cosine_similarity([s], [t])[0, 0] 
            for s, t in zip(student_proj_norm, teacher_norm)
        ])
        
        metrics['cosine_similarity_raw'] = cos_sim_raw
        metrics['cosine_similarity_projected'] = cos_sim_proj
        
        # L2 distance
        l2_dist_raw = np.mean(np.linalg.norm(student_norm - teacher_norm, axis=1))
        l2_dist_proj = np.mean(np.linalg.norm(student_proj_norm - teacher_norm, axis=1))
        
        metrics['l2_distance_raw'] = l2_dist_raw
        metrics['l2_distance_projected'] = l2_dist_proj
        
        # Centered Kernel Alignment (CKA)
        metrics['cka_raw'] = self.compute_cka(student_feats, teacher_feats)
        metrics['cka_projected'] = self.compute_cka(student_proj, teacher_feats)
        
        return metrics
    
    def compute_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Centered Kernel Alignment (CKA)"""
        
        def center_gram_matrix(K):
            n = K.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            return H @ K @ H
        
        # Linear kernel
        K_X = X @ X.T
        K_Y = Y @ Y.T
        
        # Center the matrices
        K_X = center_gram_matrix(K_X)
        K_Y = center_gram_matrix(K_Y)
        
        # Compute CKA
        numerator = np.trace(K_X @ K_Y)
        denominator = np.sqrt(np.trace(K_X @ K_X) * np.trace(K_Y @ K_Y))
        
        return numerator / (denominator + 1e-12)
    
    def plot_feature_distributions(
        self, 
        features: Dict[str, np.ndarray], 
        labels: List[int]
    ):
        """Plot feature distribution comparisons"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Feature norms
        student_norms = np.linalg.norm(features['student'], axis=1)
        teacher_norms = np.linalg.norm(features['teacher'], axis=1)
        proj_norms = np.linalg.norm(features['student_projected'], axis=1)
        
        axes[0, 0].hist(student_norms, bins=50, alpha=0.7, label='Student', color='blue')
        axes[0, 0].hist(teacher_norms, bins=50, alpha=0.7, label='Teacher', color='red')
        axes[0, 0].set_title('Feature Norm Distribution')
        axes[0, 0].set_xlabel('L2 Norm')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Projected vs Teacher norms
        axes[0, 1].hist(proj_norms, bins=50, alpha=0.7, label='Student Projected', color='green')
        axes[0, 1].hist(teacher_norms, bins=50, alpha=0.7, label='Teacher', color='red')
        axes[0, 1].set_title('Projected Feature Norm Distribution')
        axes[0, 1].set_xlabel('L2 Norm')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Cosine similarities per sample
        cos_sims = []
        for i in range(len(features['student'])):
            cos_sim = cosine_similarity(
                [features['student_projected'][i]], 
                [features['teacher'][i]]
            )[0, 0]
            cos_sims.append(cos_sim)
        
        axes[0, 2].hist(cos_sims, bins=50, alpha=0.7, color='purple')
        axes[0, 2].set_title('Per-Sample Cosine Similarity')
        axes[0, 2].set_xlabel('Cosine Similarity')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(np.mean(cos_sims), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(cos_sims):.3f}')
        axes[0, 2].legend()
        
        # Feature dimension analysis (first few dimensions)
        n_dims = min(10, features['student'].shape[1])
        dim_indices = range(n_dims)
        
        student_means = np.mean(features['student'][:, :n_dims], axis=0)
        teacher_means = np.mean(features['teacher'][:, :n_dims], axis=0)
        
        axes[1, 0].plot(dim_indices, student_means, 'o-', label='Student', color='blue')
        axes[1, 0].plot(dim_indices, teacher_means, 's-', label='Teacher', color='red')
        axes[1, 0].set_title('Feature Dimension Means')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].legend()
        
        # Class-wise similarity analysis
        unique_labels = list(set(labels))
        class_similarities = []
        
        for label in unique_labels:
            mask = np.array(labels) == label
            if np.sum(mask) > 1:  # Need at least 2 samples
                class_student = features['student_projected'][mask]
                class_teacher = features['teacher'][mask]
                
                # Average cosine similarity for this class
                class_cos_sim = np.mean([
                    cosine_similarity([s], [t])[0, 0] 
                    for s, t in zip(class_student, class_teacher)
                ])
                class_similarities.append(class_cos_sim)
            else:
                class_similarities.append(0)
        
        axes[1, 1].bar(range(len(unique_labels)), class_similarities, 
                      color='orange', alpha=0.7)
        axes[1, 1].set_title('Class-wise Cosine Similarity')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Average Cosine Similarity')
        axes[1, 1].set_xticks(range(len(unique_labels)))
        axes[1, 1].set_xticklabels(unique_labels)
        
        # Feature variance analysis
        student_var = np.var(features['student'], axis=0)
        teacher_var = np.var(features['teacher'], axis=0)
        
        axes[1, 2].scatter(teacher_var[:100], student_var[:100], alpha=0.6)
        axes[1, 2].plot([0, max(teacher_var[:100])], [0, max(teacher_var[:100])], 
                       'r--', alpha=0.8, label='y=x')
        axes[1, 2].set_title('Feature Variance Comparison (First 100 dims)')
        axes[1, 2].set_xlabel('Teacher Variance')
        axes[1, 2].set_ylabel('Student Variance')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dimensionality_reduction(
        self, 
        features: Dict[str, np.ndarray], 
        labels: List[int]
    ):
        """Plot t-SNE and UMAP visualizations"""
        
        # Sample subset for visualization if too many samples
        max_viz_samples = 1000
        if len(labels) > max_viz_samples:
            indices = np.random.choice(len(labels), max_viz_samples, replace=False)
            viz_features = {k: v[indices] for k, v in features.items()}
            viz_labels = [labels[i] for i in indices]
        else:
            viz_features = features
            viz_labels = labels
        
        # Determine figure layout based on UMAP availability
        if UMAP_AVAILABLE:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            n_rows = 2
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes = axes.reshape(1, -1)  # Make it 2D for consistent indexing
            n_rows = 1
        
        # t-SNE
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(viz_labels)//4))
        
        student_tsne = tsne.fit_transform(viz_features['student'])
        
        # Create new t-SNE instances for each feature set (sklearn requirement)
        tsne_teacher = TSNE(n_components=2, random_state=42, perplexity=min(30, len(viz_labels)//4))
        teacher_tsne = tsne_teacher.fit_transform(viz_features['teacher'])
        
        tsne_proj = TSNE(n_components=2, random_state=42, perplexity=min(30, len(viz_labels)//4))
        proj_tsne = tsne_proj.fit_transform(viz_features['student_projected'])
        
        # Plot t-SNE results
        scatter1 = axes[0, 0].scatter(student_tsne[:, 0], student_tsne[:, 1], 
                                     c=viz_labels, cmap='tab10', alpha=0.7, s=20)
        axes[0, 0].set_title('Student Features (t-SNE)')
        axes[0, 0].set_xlabel('t-SNE 1')
        axes[0, 0].set_ylabel('t-SNE 2')
        
        scatter2 = axes[0, 1].scatter(teacher_tsne[:, 0], teacher_tsne[:, 1], 
                                     c=viz_labels, cmap='tab10', alpha=0.7, s=20)
        axes[0, 1].set_title('Teacher Features (t-SNE)')
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
        
        scatter3 = axes[0, 2].scatter(proj_tsne[:, 0], proj_tsne[:, 1], 
                                     c=viz_labels, cmap='tab10', alpha=0.7, s=20)
        axes[0, 2].set_title('Student Projected Features (t-SNE)')
        axes[0, 2].set_xlabel('t-SNE 1')
        axes[0, 2].set_ylabel('t-SNE 2')
        
        # UMAP (only if available)
        if UMAP_AVAILABLE:
            print("Computing UMAP...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(viz_labels)//3))
            
            student_umap = reducer.fit_transform(viz_features['student'])
            
            reducer_teacher = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(viz_labels)//3))
            teacher_umap = reducer_teacher.fit_transform(viz_features['teacher'])
            
            reducer_proj = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(viz_labels)//3))
            proj_umap = reducer_proj.fit_transform(viz_features['student_projected'])
            
            # Plot UMAP results
            axes[1, 0].scatter(student_umap[:, 0], student_umap[:, 1], 
                              c=viz_labels, cmap='tab10', alpha=0.7, s=20)
            axes[1, 0].set_title('Student Features (UMAP)')
            axes[1, 0].set_xlabel('UMAP 1')
            axes[1, 0].set_ylabel('UMAP 2')
            
            axes[1, 1].scatter(teacher_umap[:, 0], teacher_umap[:, 1], 
                              c=viz_labels, cmap='tab10', alpha=0.7, s=20)
            axes[1, 1].set_title('Teacher Features (UMAP)')
            axes[1, 1].set_xlabel('UMAP 1')
            axes[1, 1].set_ylabel('UMAP 2')
            
            axes[1, 2].scatter(proj_umap[:, 0], proj_umap[:, 1], 
                              c=viz_labels, cmap='tab10', alpha=0.7, s=20)
            axes[1, 2].set_title('Student Projected Features (UMAP)')
            axes[1, 2].set_xlabel('UMAP 1')
            axes[1, 2].set_ylabel('UMAP 2')
        
        # Add colorbar
        if n_rows == 2:
            cbar = plt.colorbar(scatter1, ax=axes[0, :], orientation='horizontal', 
                               fraction=0.05, pad=0.1)
        else:
            cbar = plt.colorbar(scatter1, ax=axes[0, :], orientation='horizontal', 
                               fraction=0.05, pad=0.1)
        cbar.set_label('Class Label')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if not UMAP_AVAILABLE:
            print("Note: UMAP visualizations skipped. Install umap-learn for additional plots.")
    
    def plot_clustering_analysis(
        self, 
        features: Dict[str, np.ndarray], 
        labels: List[int]
    ):
        """Analyze clustering quality of features"""
        
        n_clusters = len(set(labels))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        models_to_analyze = [
            ('student', 'Student'),
            ('teacher', 'Teacher'), 
            ('student_projected', 'Student Projected')
        ]
        
        clustering_results = {}
        
        for i, (feat_key, name) in enumerate(models_to_analyze):
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features[feat_key])
            
            # Compute metrics
            ari = adjusted_rand_score(labels, cluster_labels)
            silhouette = silhouette_score(features[feat_key], cluster_labels)
            
            clustering_results[name] = {'ARI': ari, 'Silhouette': silhouette}
            
            # Plot PCA visualization with clusters
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(features[feat_key])
            
            scatter = axes[i].scatter(pca_features[:, 0], pca_features[:, 1], 
                                    c=cluster_labels, cmap='tab10', alpha=0.7, s=20)
            axes[i].set_title(f'{name}\nARI: {ari:.3f}, Silhouette: {silhouette:.3f}')
            axes[i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
            axes[i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return clustering_results
    
    def plot_similarity_heatmap(
        self, 
        features: Dict[str, np.ndarray], 
        labels: List[int],
        n_samples: int = 100
    ):
        """Plot similarity heatmaps between samples"""
        
        # Sample random subset
        indices = np.random.choice(len(labels), min(n_samples, len(labels)), replace=False)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, (feat_key, title) in enumerate([
            ('student', 'Student'),
            ('teacher', 'Teacher'),
            ('student_projected', 'Student Projected')
        ]):
            # Compute similarity matrix
            feat_subset = features[feat_key][indices]
            sim_matrix = cosine_similarity(feat_subset)
            
            # Plot heatmap
            sns.heatmap(sim_matrix, ax=axes[i], cmap='RdYlBu_r', center=0,
                       square=True, cbar_kws={'shrink': 0.8})
            axes[i].set_title(f'{title} Similarity Matrix')
            axes[i].set_xlabel('Sample Index')
            axes[i].set_ylabel('Sample Index')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/similarity_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(
        self,
        dataloader,
        max_samples: int = 1000
    ):
        """Generate comprehensive evaluation report"""
        
        print("="*60)
        print("DINOV2 DISTILLATION EVALUATION REPORT")
        print("="*60)
        
        # Extract features
        features, labels = self.extract_features(dataloader, max_samples)
        
        # Compute similarity metrics
        print("\n1. Computing similarity metrics...")
        metrics = self.compute_similarity_metrics(features)
        
        print("\nSimilarity Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Generate visualizations
        print("\n2. Generating feature distribution plots...")
        self.plot_feature_distributions(features, labels)
        
        print("\n3. Generating dimensionality reduction plots...")
        self.plot_dimensionality_reduction(features, labels)
        
        print("\n4. Analyzing clustering quality...")
        clustering_results = self.plot_clustering_analysis(features, labels)
        
        print("\nClustering Results:")
        print("-" * 30)
        for model, results in clustering_results.items():
            print(f"{model} - ARI: {results['ARI']:.4f}, Silhouette: {results['Silhouette']:.4f}")
        
        print("\n5. Generating similarity heatmaps...")
        self.plot_similarity_heatmap(features, labels)
        
        # Save metrics to file
        results_summary = {
            'similarity_metrics': metrics,
            'clustering_results': clustering_results,
            'num_samples': len(labels),
            'num_classes': len(set(labels))
        }
        
        import json
        with open(f'{self.save_dir}/evaluation_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n6. Evaluation complete! Results saved to {self.save_dir}")
        print("="*60)
        
        return results_summary

# Usage example
def main():
    # Configuration (should match your training configuration)
    student_config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.0,
    }
    
    # Import your dataloader
    from data.hf_cifar import get_cifar_dataloader  # Replace with your actual import
    
    # Get test dataloader
    test_loader = get_cifar_dataloader(
        split="test",  # or "train[90%:]" for a held-out set
        batch_size=64,
        shuffle=False
    )
    
    # Initialize evaluator
    evaluator = DistillationEvaluator(
        student_model_path="enhanced_dinov2_student_cifar10.pth",  # Path to your trained model
        student_config=student_config,
        teacher_model_name="facebook/dinov2-small",
        save_dir="evaluation_results"
    )
    
    # Run comprehensive evaluation
    results = evaluator.generate_evaluation_report(
        dataloader=test_loader,
        max_samples=1000  # Adjust based on your needs
    )
    
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()