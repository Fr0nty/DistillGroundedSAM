#!/usr/bin/env python3
"""
Quick evaluation runner for DINOv2 distillation results
This script provides easy-to-use functions for evaluating your trained models
"""

import os
import sys
import torch
from pathlib import Path

def quick_evaluate(
    student_model_path: str,
    dataset_choice: str = "food101",
    num_batches: int = 10,
    num_samples_viz: int = 8,
    output_dir: str = None
):
    """
    Quick evaluation with minimal setup
    
    Args:
        student_model_path: Path to your trained student model
        dataset_choice: 'food101', 'stl10', 'oxford_pets', 'flowers102', or 'imagenette'
        num_batches: Number of batches to evaluate (more = better stats, slower)
        num_samples_viz: Number of samples to show in visualizations
        output_dir: Where to save results (auto-generated if None)
    """
    
    print("🚀 Quick DINOv2 Distillation Evaluation")
    print("="*40)
    
    # Auto-generate output directory if not provided
    if output_dir is None:
        model_name = Path(student_model_path).stem
        output_dir = f"eval_{model_name}_{dataset_choice}"
    
    # Import the evaluation classes
    try:
        from eval_enhanced import ComprehensiveEvaluator
        from data.multiple_datasets import (
            get_food101_dataloader, get_stl10_dataloader, 
            get_oxford_pets_dataloader, get_flowers102_dataloader,
            get_imagenette_dataloader
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the evaluation scripts are in the same directory")
        return
    
    # Student configuration (adjust if needed)
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
    print(f"🔧 Setting up evaluation with output dir: {output_dir}")
    evaluator = ComprehensiveEvaluator(
        student_model_path=student_model_path,
        student_config=student_config,
        save_dir=output_dir
    )
    
    # Load appropriate dataset
    print(f"📚 Loading {dataset_choice} dataset...")
    dataset_loaders = {
        'food101': lambda: get_food101_dataloader(split="validation[:5%]", batch_size=16),
        'stl10': lambda: get_stl10_dataloader(split="test", batch_size=16),
        'oxford_pets': lambda: get_oxford_pets_dataloader(split="test", batch_size=16),
        'flowers102': lambda: get_flowers102_dataloader(split="test", batch_size=16),
        'imagenette': lambda: get_imagenette_dataloader(split="validation", batch_size=16),
    }
    
    if dataset_choice not in dataset_loaders:
        print(f"❌ Unknown dataset: {dataset_choice}")
        print(f"Available: {list(dataset_loaders.keys())}")
        return
    
    test_loader = dataset_loaders[dataset_choice]()
    
    # Run evaluation
    print(f"🔍 Extracting features from {num_batches} batches...")
    evaluator.extract_comprehensive_features(test_loader, max_batches=num_batches)
    
    print("📊 Generating visualizations...")
    evaluator.plot_similarity_analysis()
    evaluator.plot_feature_space_comparison()
    evaluator.visualize_sample_comparisons(num_samples=num_samples_viz)
    evaluator.plot_attention_analysis()
    
    # Generate report
    metrics, report = evaluator.generate_evaluation_report()
    
    print(f"\n✅ Evaluation complete! Check '{output_dir}/' for results")
    print(f"📈 Mean similarity: {metrics['mean_similarity']:.3f}")
    
    return metrics, output_dir

def compare_models(model_paths: list, dataset_choice: str = "food101"):
    """
    Compare multiple student models side by side
    
    Args:
        model_paths: List of paths to different student models
        dataset_choice: Dataset to use for comparison
    """
    
    print("🔀 Comparing Multiple Models")
    print("="*30)
    
    results = {}
    
    for i, model_path in enumerate(model_paths):
        print(f"\n📊 Evaluating model {i+1}/{len(model_paths)}: {model_path}")
        
        try:
            metrics, output_dir = quick_evaluate(
                student_model_path=model_path,
                dataset_choice=dataset_choice,
                num_batches=5,  # Fewer batches for comparison
                num_samples_viz=4,
                output_dir=f"comparison_model_{i+1}"
            )
            
            results[model_path] = metrics
            
        except Exception as e:
            print(f"❌ Error evaluating {model_path}: {e}")
            results[model_path] = None
    
    # Create comparison summary
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'Mean Sim':<10} {'Good %':<10} {'Status'}")
    print("-"*60)
    
    for model_path, metrics in results.items():
        if metrics:
            good_percent = ((metrics['samples_very_good'] + metrics['samples_excellent']) / 
                          metrics['samples_total']) * 100
            
            status = "🟢 Excellent" if metrics['mean_similarity'] > 0.8 else \
                    "🟡 Good" if metrics['mean_similarity'] > 0.7 else \
                    "🟠 Moderate" if metrics['mean_similarity'] > 0.6 else "🔴 Poor"
            
            model_name = Path(model_path).name[:28]
            print(f"{model_name:<30} {metrics['mean_similarity']:<10.3f} {good_percent:<10.1f} {status}")
        else:
            model_name = Path(model_path).name[:28]
            print(f"{model_name:<30} {'ERROR':<10} {'N/A':<10} ❌ Failed")
    
    print("="*60)
    
    return results

def dataset_benchmark(student_model_path: str, datasets: list = None):
    """
    Test the same model on different datasets to see performance consistency
    
    Args:
        student_model_path: Path to student model
        datasets: List of datasets to test on
    """
    
    if datasets is None:
        datasets = ['food101', 'stl10', 'oxford_pets']
    
    print("🎯 Dataset Benchmark")
    print("="*25)
    
    results = {}
    
    for dataset in datasets:
        print(f"\n📊 Testing on {dataset}...")
        
        try:
            metrics, output_dir = quick_evaluate(
                student_model_path=student_model_path,
                dataset_choice=dataset,
                num_batches=8,
                num_samples_viz=6,
                output_dir=f"benchmark_{dataset}"
            )
            
            results[dataset] = metrics
            
        except Exception as e:
            print(f"❌ Error with {dataset}: {e}")
            results[dataset] = None
    
    # Summary
    print("\n" + "="*50)
    print("📊 DATASET BENCHMARK SUMMARY")
    print("="*50)
    print(f"{'Dataset':<15} {'Mean Sim':<10} {'Std':<8} {'Status'}")
    print("-"*50)
    
    for dataset, metrics in results.items():
        if metrics:
            status = "🟢 Great" if metrics['mean_similarity'] > 0.8 else \
                    "🟡 Good" if metrics['mean_similarity'] > 0.7 else \
                    "🟠 OK" if metrics['mean_similarity'] > 0.6 else "🔴 Poor"
            
            print(f"{dataset:<15} {metrics['mean_similarity']:<10.3f} {metrics['std_similarity']:<8.3f} {status}")
        else:
            print(f"{dataset:<15} {'ERROR':<10} {'N/A':<8} ❌")
    
    print("="*50)
    
    return results

def main():
    """Interactive main function"""
    
    print("🎨 DINOv2 Distillation Evaluation Suite")
    print("="*40)
    
    # Check if we have command line arguments
    if len(sys.argv) > 1:
        student_model_path = sys.argv[1]
        dataset_choice = sys.argv[2] if len(sys.argv) > 2 else "food101"
        
        print(f"Running evaluation on: {student_model_path}")
        quick_evaluate(student_model_path, dataset_choice)
        return
    
    # Interactive mode
    print("Choose evaluation type:")
    print("1. Single model evaluation")
    print("2. Compare multiple models")
    print("3. Dataset benchmark")
    print("4. Quick demo")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        model_path = input("Enter student model path: ").strip()
        dataset = input("Dataset (food101/stl10/oxford_pets/flowers102/imagenette): ").strip() or "food101"
        quick_evaluate(model_path, dataset)
        
    elif choice == "2":
        print("Enter model paths (one per line, empty line to finish):")
        model_paths = []
        while True:
            path = input().strip()
            if not path:
                break
            model_paths.append(path)
        
        if model_paths:
            dataset = input("Dataset to use: ").strip() or "food101"
            compare_models(model_paths, dataset)
        else:
            print("No model paths provided")
            
    elif choice == "3":
        model_path = input("Enter student model path: ").strip()
        dataset_benchmark(model_path)
        
    elif choice == "4":
        # Demo with example paths
        demo_models = [
            "enhanced_dinov2_student_cifar10.pth",
            "enhanced_dinov2_student_food101.pth",
            
        ]
        
        print("Demo mode - testing example model paths...")
        for model in demo_models:
            if os.path.exists(model):
                print(f"Found model: {model}")
                quick_evaluate(model, "food101", num_batches=3, num_samples_viz=4)
                break
        else:
            print("No demo models found. Please run with your own model path.")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()