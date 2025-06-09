# evaluation_system.py
"""
Independent Evaluation System for GroundingDINO Distillation
Generates comprehensive evaluation plots and metrics for trained models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import pickle
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Try importing analysis tools
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    print("Warning: thop not available. Install with: pip install thop")
    THOP_AVAILABLE = False

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    print("Warning: ptflops not available. Install with: pip install ptflops")
    PTFLOPS_AVAILABLE = False

from sklearn.metrics import average_precision_score, precision_recall_curve
from torchvision import transforms
from torch.utils.data import DataLoader
import timm

# Import your modules (adjust paths as needed)
try:
    from GrdDINO_Distill import MultiStudentArchitecture, DistillationConfig
    from data.multiple_datasets import (
        get_oxford_pets_dataloader, get_food101_dataloader, 
        get_stl10_dataloader, get_flowers102_dataloader, get_imagenette_dataloader
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")
    MODULES_AVAILABLE = False


class ModelMetrics:
    """Compute comprehensive model metrics."""
    
    def __init__(self, model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
        self.model = model
        self.input_size = input_size
        self.device = next(model.parameters()).device
        
    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_parameters_M': total_params / 1e6,
            'trainable_parameters_M': trainable_params / 1e6
        }
    
    def compute_flops(self) -> Dict[str, float]:
        """Compute FLOPs using available libraries."""
        flops_info = {}
        
        # Create dummy input
        dummy_input = torch.randn(1, *self.input_size).to(self.device)
        
        # Method 1: thop
        if THOP_AVAILABLE:
            try:
                model_copy = self.model.eval()
                flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
                flops_info['thop_flops'] = flops
                flops_info['thop_flops_G'] = flops / 1e9
                flops_info['thop_params'] = params
            except Exception as e:
                print(f"Warning: thop FLOPs computation failed: {e}")
        
        # Method 2: ptflops
        if PTFLOPS_AVAILABLE:
            try:
                self.model.eval()
                flops, params = get_model_complexity_info(
                    self.model, 
                    self.input_size, 
                    print_per_layer_stat=False,
                    as_strings=False
                )
                flops_info['ptflops_flops'] = flops
                flops_info['ptflops_flops_G'] = flops / 1e9
                flops_info['ptflops_params'] = params
            except Exception as e:
                print(f"Warning: ptflops FLOPs computation failed: {e}")
        
        return flops_info
    
    def measure_inference_speed(self, batch_sizes: List[int] = [1, 8, 16], 
                              num_runs: int = 100) -> Dict[str, float]:
        """Measure inference speed for different batch sizes."""
        self.model.eval()
        speed_metrics = {}
        
        with torch.no_grad():
            for batch_size in batch_sizes:
                dummy_input = torch.randn(batch_size, *self.input_size).to(self.device)
                
                # Warmup
                for _ in range(10):
                    _ = self.model(dummy_input)
                
                # Measure
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                start_time = time.time()
                
                for _ in range(num_runs):
                    _ = self.model(dummy_input)
                
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.time()
                
                total_time = end_time - start_time
                avg_time_per_batch = total_time / num_runs
                avg_time_per_sample = avg_time_per_batch / batch_size
                fps = 1.0 / avg_time_per_sample
                
                speed_metrics[f'batch_{batch_size}_time_ms'] = avg_time_per_batch * 1000
                speed_metrics[f'batch_{batch_size}_fps'] = fps
                speed_metrics[f'batch_{batch_size}_latency_ms'] = avg_time_per_sample * 1000
        
        return speed_metrics
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb


class DistillationEvaluator:
    """Comprehensive evaluation of distilled models."""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.config = self._load_config()
        self.history = self._load_history()
        
    def _load_config(self) -> Optional[Dict]:
        """Load experiment configuration."""
        config_path = self.experiment_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return None
    
    def _load_history(self) -> Optional[Dict]:
        """Load training history."""
        history_path = self.experiment_dir / "training_history.pkl"
        if history_path.exists():
            with open(history_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def load_best_model(self) -> Optional[nn.Module]:
        """Load the best trained model."""
        if not MODULES_AVAILABLE:
            print("Error: Cannot load model - project modules not available")
            return None
            
        best_model_path = self.experiment_dir / "best_model.pth"
        if not best_model_path.exists():
            print(f"Error: Best model not found: {best_model_path}")
            return None
        
        try:
            # Create model from config
            if self.config:
                model = MultiStudentArchitecture(
                    student_type=self.config['student_type'],
                    model_name=self.config['student_model'],
                    num_classes=self.config['num_classes'],
                    pretrained=False  # We're loading trained weights
                )
                
                # Load checkpoint - fix for PyTorch 2.6+ security
                checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['student_state_dict'])
                model.eval()
                
                print(f"Loaded model: {self.config['student_type']} ({self.config['student_model']})")
                return model
            else:
                print("Error: No config found, cannot create model")
                return None
                
        except Exception as e:
            print(f"Error: Failed to load model: {e}")
            return None
    
    def compute_model_metrics(self, model: nn.Module) -> Dict[str, any]:
        """Compute comprehensive model metrics."""
        if model is None:
            return {}
        
        metrics_computer = ModelMetrics(model)
        
        metrics = {}
        
        # Parameter count
        param_metrics = metrics_computer.count_parameters()
        metrics.update(param_metrics)
        
        # FLOPs
        flop_metrics = metrics_computer.compute_flops()
        metrics.update(flop_metrics)
        
        # Inference speed
        speed_metrics = metrics_computer.measure_inference_speed()
        metrics.update(speed_metrics)
        
        # Model size
        metrics['model_size_mb'] = metrics_computer.get_model_size_mb()
        
        return metrics
    
    def evaluate_accuracy(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model accuracy and compute mAP."""
        if model is None or test_loader is None:
            return {}
        
        model.eval()
        device = next(model.parameters()).device
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                if batch_idx % 50 == 0:
                    print(f"Evaluating batch {batch_idx}/{len(test_loader)}")
                
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Predictions
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                # Accumulate
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        
        # Compute mAP
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # One-hot encode labels for mAP calculation
        num_classes = all_probabilities.shape[1]
        labels_onehot = np.eye(num_classes)[all_labels]
        
        # Compute AP for each class
        aps = []
        for i in range(num_classes):
            if np.sum(labels_onehot[:, i]) > 0:  # Only if class exists in test set
                ap = average_precision_score(labels_onehot[:, i], all_probabilities[:, i])
                aps.append(ap)
        
        mean_ap = np.mean(aps) if aps else 0.0
        
        return {
            'accuracy': accuracy,
            'mean_average_precision': mean_ap * 100,  # Convert to percentage
            'num_samples': total,
            'num_classes_evaluated': len(aps)
        }
    
    def create_test_loader(self) -> Optional[DataLoader]:
        """Create test data loader based on experiment config."""
        if not self.config or not MODULES_AVAILABLE:
            return None
        
        dataset_name = self.config['dataset_name']
        
        # Map dataset names to loader functions
        loader_map = {
            'Oxford Pets': get_oxford_pets_dataloader,
            'Food101': get_food101_dataloader,
            'Food101 Full': get_food101_dataloader,
            'STL10': get_stl10_dataloader,
            'Flowers102': get_flowers102_dataloader,
            'Imagenette': get_imagenette_dataloader
        }
        
        if dataset_name not in loader_map:
            print(f"Warning: Unknown dataset: {dataset_name}")
            return None
        
        try:
            loader_func = loader_map[dataset_name]
            
            # Use test split
            test_split = "test" if dataset_name not in ["Food101", "Food101 Full"] else "validation"
            if dataset_name in ["Food101", "Food101 Full"]:
                test_split = "validation[:20%]"  # Subset for speed
            
            test_loader = loader_func(
                split=test_split,
                batch_size=32,
                shuffle=False,
                num_workers=2
            )
            
            print(f"Created test loader for {dataset_name}")
            return test_loader
            
        except Exception as e:
            print(f"Error: Failed to create test loader: {e}")
            return None


class EvaluationPlotter:
    """Create evaluation plots for distillation experiments."""
    
    def __init__(self, output_dir: str = "evaluation_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self, history: Dict, experiment_name: str):
        """Plot training and validation loss curves."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Training Curves - {experiment_name}', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Total Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Component Losses
        axes[0, 1].plot(epochs, history['task_loss'], 'g-', label='Task Loss', linewidth=2)
        axes[0, 1].plot(epochs, history['feature_loss'], 'orange', label='Feature Loss', linewidth=2)
        axes[0, 1].plot(epochs, history['attention_loss'], 'purple', label='Attention Loss', linewidth=2)
        axes[0, 1].set_title('Component Losses', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Validation Accuracy
        axes[1, 0].plot(epochs, history['val_accuracy'], 'darkgreen', linewidth=3)
        axes[1, 0].set_title('Validation Accuracy', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].fill_between(epochs, history['val_accuracy'], alpha=0.3, color='darkgreen')
        
        # Learning Rate
        axes[1, 1].plot(epochs, history['learning_rates'], 'crimson', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{experiment_name}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved: {experiment_name}_training_curves.png")
    
    def plot_model_comparison(self, experiments_data: List[Dict]):
        """Plot distillation-specific comparison charts."""
        
        if not experiments_data:
            print("Warning: No experiment data for comparison")
            return
        
        # Create DataFrame
        df = pd.DataFrame(experiments_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Distillation Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Knowledge Transfer Efficiency (Feature Loss vs Final Accuracy)
        if 'final_train_loss' in df.columns and 'accuracy' in df.columns:
            # Calculate knowledge transfer efficiency
            df['knowledge_transfer_efficiency'] = df['accuracy'] / (df['final_train_loss'] + 1e-8)
            
            bars = axes[0, 0].bar(range(len(df)), df['knowledge_transfer_efficiency'], 
                                alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(df))))
            axes[0, 0].set_xlabel('Experiments')
            axes[0, 0].set_ylabel('Knowledge Transfer Efficiency')
            axes[0, 0].set_title('Knowledge Transfer Efficiency\n(Accuracy / Final Loss)', fontweight='bold')
            axes[0, 0].set_xticks(range(len(df)))
            axes[0, 0].set_xticklabels(df['experiment_name'], rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, df['knowledge_transfer_efficiency']):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Training Convergence Speed (Best accuracy and epochs to reach it)
        if 'best_val_accuracy' in df.columns:
            x_pos = np.arange(len(df))
            width = 0.35
            
            bars1 = axes[0, 1].bar(x_pos - width/2, df['best_val_accuracy'], width,
                                  label='Best Accuracy (%)', alpha=0.8, color='skyblue')
            
            # Add secondary y-axis for epochs if available (placeholder for now)
            axes[0, 1].set_xlabel('Experiments')
            axes[0, 1].set_ylabel('Best Validation Accuracy (%)')
            axes[0, 1].set_title('Training Performance', fontweight='bold')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(df['experiment_name'], rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars1, df['best_val_accuracy']):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Model Efficiency Score (Accuracy per Parameter)
        if 'accuracy' in df.columns and 'total_parameters_M' in df.columns:
            df['efficiency_score'] = df['accuracy'] / df['total_parameters_M']
            
            scatter = axes[1, 0].scatter(df['total_parameters_M'], df['accuracy'], 
                                       s=150, alpha=0.7, c=df['efficiency_score'], 
                                       cmap='plasma', edgecolors='black')
            
            for i, row in df.iterrows():
                axes[1, 0].annotate(f"{row['efficiency_score']:.1f}", 
                                  (row['total_parameters_M'], row['accuracy']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            axes[1, 0].set_xlabel('Model Size (Million Parameters)')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_title('Model Efficiency\n(Color = Accuracy/Param Ratio)', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[1, 0])
            cbar.set_label('Efficiency Score')
        
        # 4. Distillation Loss Balance (if available)
        if all(col in df.columns for col in ['experiment_name']):
            # For now, show a summary of available metrics
            available_metrics = [col for col in df.columns if col not in ['experiment_name', 'config']]
            
            # Create a radar-like plot showing relative performance
            if len(available_metrics) >= 3:
                # Select top 3 numeric metrics
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                top_metrics = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                
                if len(top_metrics) > 0:
                    # Normalize metrics to 0-100 scale
                    normalized_data = df[top_metrics].copy()
                    for col in top_metrics:
                        min_val = normalized_data[col].min()
                        max_val = normalized_data[col].max()
                        if max_val != min_val:
                            normalized_data[col] = ((normalized_data[col] - min_val) / (max_val - min_val)) * 100
                    
                    x_pos = np.arange(len(df))
                    width = 0.25
                    
                    for i, metric in enumerate(top_metrics):
                        axes[1, 1].bar(x_pos + i*width, normalized_data[metric], width,
                                     label=metric, alpha=0.8)
                    
                    axes[1, 1].set_xlabel('Experiments')
                    axes[1, 1].set_ylabel('Normalized Score (0-100)')
                    axes[1, 1].set_title('Multi-Metric Performance', fontweight='bold')
                    axes[1, 1].set_xticks(x_pos + width)
                    axes[1, 1].set_xticklabels(df['experiment_name'], rotation=45, ha='right')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distillation_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Distillation performance analysis saved: distillation_performance_analysis.png")
    
    def plot_efficiency_frontier(self, experiments_data: List[Dict]):
        """Plot distillation-specific efficiency analysis."""
        
        if not experiments_data:
            return
        
        df = pd.DataFrame(experiments_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Distillation Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # 1. Training Efficiency (Accuracy vs Training Time/Epochs)
        if 'accuracy' in df.columns:
            # Use number of training epochs as a proxy for training time
            # Assume 20 epochs as default if not available
            training_effort = [20] * len(df)  # Default value
            
            axes[0].scatter(training_effort, df['accuracy'], 
                          s=150, alpha=0.7, c=range(len(df)), cmap='viridis', edgecolors='black')
            
            for i, row in df.iterrows():
                axes[0].annotate(row['experiment_name'], 
                               (training_effort[i], row['accuracy']),
                               xytext=(10, 10), textcoords='offset points', 
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            axes[0].set_xlabel('Training Effort (Epochs)', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Final Accuracy (%)', fontsize=12, fontweight='bold')
            axes[0].set_title('Training Efficiency\n(Upper Left = Most Efficient)', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
        
        # 2. Knowledge Distillation Quality vs Model Complexity
        if 'accuracy' in df.columns and 'total_parameters_M' in df.columns:
            # Calculate distillation quality score
            # Higher accuracy with fewer parameters = better distillation
            max_acc = df['accuracy'].max()
            min_acc = df['accuracy'].min()
            max_params = df['total_parameters_M'].max()
            min_params = df['total_parameters_M'].min()
            
            # Normalize and create composite score
            acc_norm = (df['accuracy'] - min_acc) / (max_acc - min_acc + 1e-8)
            param_norm = 1 - (df['total_parameters_M'] - min_params) / (max_params - min_params + 1e-8)  # Inverted (smaller is better)
            
            distillation_quality = (acc_norm + param_norm) / 2 * 100  # Convert to percentage
            
            scatter = axes[1].scatter(df['total_parameters_M'], df['accuracy'], 
                          s=200, alpha=0.7, c=distillation_quality, cmap='plasma', edgecolors='black')
            
            for i, row in df.iterrows():
                axes[1].annotate(f"Q: {distillation_quality.iloc[i]:.1f}", 
                               (row['total_parameters_M'], row['accuracy']),
                               xytext=(10, 10), textcoords='offset points', 
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            
            axes[1].set_xlabel('Model Size (Million Parameters)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            axes[1].set_title('Distillation Quality\n(Upper Left = Best Quality)', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[1])
            cbar.set_label('Distillation Quality Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distillation_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Distillation efficiency analysis saved: distillation_efficiency_analysis.png")


def evaluate_single_experiment(experiment_dir: str) -> Dict[str, any]:
    """Evaluate a single experiment comprehensively."""
    
    print(f"\nEvaluating experiment: {experiment_dir}")
    print("=" * 60)
    
    evaluator = DistillationEvaluator(experiment_dir)
    
    if not evaluator.config:
        print("Error: No config found, skipping")
        return {}
    
    # Load model
    model = evaluator.load_best_model()
    if model is None:
        print("Error: Could not load model, skipping")
        return {}
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    results = {
        'experiment_name': Path(experiment_dir).name,
        'config': evaluator.config
    }
    
    # Compute model metrics
    print("Computing model metrics...")
    model_metrics = evaluator.compute_model_metrics(model)
    results.update(model_metrics)
    
    # Evaluate accuracy
    print("Evaluating accuracy...")
    test_loader = evaluator.create_test_loader()
    if test_loader:
        accuracy_metrics = evaluator.evaluate_accuracy(model, test_loader)
        results.update(accuracy_metrics)
    else:
        print("Warning: Could not create test loader")
    
    # Add training history metrics
    if evaluator.history:
        results['best_val_accuracy'] = max(evaluator.history['val_accuracy'])
        results['final_train_loss'] = evaluator.history['train_loss'][-1]
        results['final_val_loss'] = evaluator.history['val_loss'][-1]
    
    print(f"Evaluation complete for {results['experiment_name']}")
    return results


def evaluate_multiple_experiments(experiments_base_dir: str = "experiments"):
    """Evaluate all experiments in a directory."""
    
    print("Multi-Experiment Evaluation System")
    print("=" * 70)
    
    base_path = Path(experiments_base_dir)
    if not base_path.exists():
        print(f"Error: Experiments directory not found: {experiments_base_dir}")
        return
    
    # Find all experiment directories
    experiment_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not experiment_dirs:
        print(f"Error: No experiment directories found in {experiments_base_dir}")
        return
    
    print(f"Found {len(experiment_dirs)} experiments:")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir.name}")
    
    # Evaluate each experiment
    all_results = []
    plotter = EvaluationPlotter()
    
    for exp_dir in experiment_dirs:
        try:
            result = evaluate_single_experiment(str(exp_dir))
            if result:
                all_results.append(result)
                
                # Plot training curves if history available
                evaluator = DistillationEvaluator(str(exp_dir))
                if evaluator.history:
                    plotter.plot_training_curves(evaluator.history, exp_dir.name)
                    
        except Exception as e:
            print(f"Error: Failed to evaluate {exp_dir.name}: {e}")
    
    if not all_results:
        print("Error: No successful evaluations")
        return
    
    # Create comparison plots
    print(f"\nCreating comparison plots for {len(all_results)} experiments...")
    plotter.plot_model_comparison(all_results)
    plotter.plot_efficiency_frontier(all_results)
    
    # Save results summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(plotter.output_dir / 'evaluation_summary.csv', index=False)
    
    # Print summary
    print(f"\nEVALUATION SUMMARY")
    print("=" * 70)
    
    if 'accuracy' in results_df.columns:
        print("Top 3 Models by Accuracy:")
        top_acc = results_df.nlargest(3, 'accuracy')[['experiment_name', 'accuracy', 'total_parameters_M']]
        for i, (_, row) in enumerate(top_acc.iterrows(), 1):
            print(f"  {i}. {row['experiment_name']}: {row['accuracy']:.2f}% ({row['total_parameters_M']:.1f}M params)")
    
    if 'batch_1_fps' in results_df.columns:
        print("\nTop 3 Models by Speed:")
        top_speed = results_df.nlargest(3, 'batch_1_fps')[['experiment_name', 'batch_1_fps', 'accuracy']]
        for i, (_, row) in enumerate(top_speed.iterrows(), 1):
            print(f"  {i}. {row['experiment_name']}: {row['batch_1_fps']:.1f} FPS ({row['accuracy']:.2f}% acc)")
    
    if 'total_parameters_M' in results_df.columns:
        print("\nTop 3 Most Efficient Models:")
        # Efficiency score: accuracy per million parameters
        results_df['efficiency'] = results_df.get('accuracy', 0) / results_df['total_parameters_M']
        top_eff = results_df.nlargest(3, 'efficiency')[['experiment_name', 'efficiency', 'accuracy', 'total_parameters_M']]
        for i, (_, row) in enumerate(top_eff.iterrows(), 1):
            print(f"  {i}. {row['experiment_name']}: {row['efficiency']:.2f} acc/M-param ({row['accuracy']:.2f}%, {row['total_parameters_M']:.1f}M)")
    
    print(f"\nAll plots saved in: {plotter.output_dir}")
    print(f"Summary CSV saved: evaluation_summary.csv")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "single" and len(sys.argv) > 2:
            # Evaluate single experiment
            experiment_dir = sys.argv[2]
            result = evaluate_single_experiment(experiment_dir)
            print(f"\nResults: {result}")
            
        elif sys.argv[1] == "multi":
            # Evaluate all experiments
            experiments_dir = sys.argv[2] if len(sys.argv) > 2 else "experiments"
            evaluate_multiple_experiments(experiments_dir)
            
        else:
            print("Usage:")
            print("  python evaluation_system.py single <experiment_dir>")
            print("  python evaluation_system.py multi [experiments_base_dir]")
    else:
        # Default: evaluate all experiments
        evaluate_multiple_experiments()