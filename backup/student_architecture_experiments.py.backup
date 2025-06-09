# student_architecture_experiments.py
"""
GroundingDINO → Multiple Student Architecture Experiments
Focus on comparing different student models with GroundingDINO teacher.
"""

from GrdDINO_Distill import DistillationConfig, GroundingDINODistillationTrainer
from data.multiple_datasets import (
    get_oxford_pets_dataloader, 
    get_food101_dataloader,
    get_stl10_dataloader,
    get_flowers102_dataloader,
    get_imagenette_dataloader
)


def get_student_architectures():
    """
    Get all available student architecture configurations.
    
    Returns:
        Dictionary of student architecture configs
    """
    
    architectures = {
        
        # ===== VISION TRANSFORMERS =====
        
        "vit_base": {
            "student_type": "vit",
            "student_model": "vit_base_patch16_224",
            "description": "Standard Vision Transformer - baseline",
            "params": "86M",
            "speed": "Fast",
            "recommendation": "🥇 BEST BASELINE"
        },
        
        "vit_small": {
            "student_type": "vit", 
            "student_model": "vit_small_patch16_224",
            "description": "Smaller ViT - faster inference",
            "params": "22M",
            "speed": "Very Fast",
            "recommendation": "⚡ SPEED FOCUSED"
        },
        
        "vit_large": {
            "student_type": "vit",
            "student_model": "vit_large_patch16_224", 
            "description": "Large ViT - better capacity",
            "params": "307M",
            "speed": "Slow",
            "recommendation": "🎯 ACCURACY FOCUSED"
        },
        
        # ===== MASKED AUTOENCODERS =====
        
        "mae_base": {
            "student_type": "mae",
            "student_model": "mae_base",
            "description": "MAE pretrained ViT - excellent representations",
            "params": "86M",
            "speed": "Fast",
            "recommendation": "🏆 EXCELLENT CHOICE"
        },
        
        "mae_large": {
            "student_type": "mae",
            "student_model": "mae_large", 
            "description": "Large MAE - state-of-the-art features",
            "params": "307M",
            "speed": "Slow",
            "recommendation": "👑 BEST ACCURACY"
        },
        
        # ===== DISTILLED TRANSFORMERS =====
        
        "deit_base": {
            "student_type": "deit",
            "student_model": "deit_base",
            "description": "DeiT - already distilled, efficient",
            "params": "86M", 
            "speed": "Fast",
            "recommendation": "🎓 DISTILLATION EXPERT"
        },
        
        "deit_small": {
            "student_type": "deit",
            "student_model": "deit_small",
            "description": "Small DeiT - very efficient",
            "params": "22M",
            "speed": "Very Fast", 
            "recommendation": "💨 EFFICIENCY KING"
        },
        
        # ===== HIERARCHICAL TRANSFORMERS =====
        
        "swin_base": {
            "student_type": "swin",
            "student_model": "swin_base",
            "description": "Swin Transformer - hierarchical features",
            "params": "88M",
            "speed": "Medium",
            "recommendation": "🏗️ MULTI-SCALE"
        },
        
        "swin_small": {
            "student_type": "swin", 
            "student_model": "swin_small",
            "description": "Small Swin - efficient hierarchical",
            "params": "50M",
            "speed": "Fast",
            "recommendation": "⚖️ BALANCED"
        },
        
        # ===== CONVOLUTIONAL NETWORKS =====
        
        "convnext_base": {
            "student_type": "convnext",
            "student_model": "convnext_base",
            "description": "ConvNeXt - modern ConvNet design",
            "params": "89M",
            "speed": "Fast",
            "recommendation": "🔄 CONV COMEBACK"
        },
        
        "convnext_small": {
            "student_type": "convnext",
            "student_model": "convnext_small", 
            "description": "Small ConvNeXt - efficient convolution",
            "params": "50M",
            "speed": "Very Fast",
            "recommendation": "🚀 CONV SPEED"
        }
    }
    
    return architectures


def get_dataset_configs():
    """Get optimized dataset configurations for GroundingDINO distillation."""
    
    configs = {
        
        # ===== HIGH QUALITY DATASETS (RECOMMENDED) =====
        
        "oxford_pets": {
            "dataset_name": "Oxford Pets",
            "num_classes": 37,
            "loader_function": get_oxford_pets_dataloader,
            "train_split": "train",
            "val_split": "test",
            "batch_size": 16,
            "description": "37 pet breeds - perfect for object detection distillation",
            "grounding_benefit": "🎯 EXCELLENT - clear object boundaries"
        },
        
        "food101": {
            "dataset_name": "Food101",
            "num_classes": 101,
            "loader_function": get_food101_dataloader,
            "train_split": "train[:30%]",  # Larger subset for better results
            "val_split": "validation[:50%]",
            "batch_size": 12,
            "description": "101 food categories - complex objects",
            "grounding_benefit": "🍕 GREAT - complex food objects"
        },
        
        "food101_full": {
            "dataset_name": "Food101 Full",
            "num_classes": 101,
            "loader_function": get_food101_dataloader,
            "train_split": "train",
            "val_split": "validation", 
            "batch_size": 8,
            "description": "Full Food101 - maximum data",
            "grounding_benefit": "👑 MAXIMUM - full dataset power"
        },
        
        "flowers102": {
            "dataset_name": "Flowers102",
            "num_classes": 102,
            "loader_function": get_flowers102_dataloader,
            "train_split": "train",
            "val_split": "test",
            "batch_size": 12,
            "description": "102 flower species - fine-grained classification",
            "grounding_benefit": "🌸 EXCELLENT - clear flower objects"
        },
        
        "imagenette": {
            "dataset_name": "Imagenette", 
            "num_classes": 10,
            "loader_function": get_imagenette_dataloader,
            "train_split": "train",
            "val_split": "validation",
            "batch_size": 20,
            "description": "10 ImageNet classes - highest quality",
            "grounding_benefit": "⭐ PERFECT - ImageNet quality objects"
        },
        
        # ===== MEDIUM QUALITY DATASETS =====
        
        "stl10": {
            "dataset_name": "STL10",
            "num_classes": 10,
            "loader_function": get_stl10_dataloader,
            "train_split": "train", 
            "val_split": "test",
            "batch_size": 24,
            "description": "10 classes, 96x96 resolution",
            "grounding_benefit": "👍 GOOD - decent resolution"
        },
        

    }
    
    return configs


def create_experiment_config(dataset_name: str, 
                           student_arch: str,
                           num_epochs: int = 25,
                           experiment_name: str = None) -> DistillationConfig:
    """
    Create a complete experiment configuration.
    
    Args:
        dataset_name: Dataset to use
        student_arch: Student architecture to use
        num_epochs: Number of training epochs
        experiment_name: Custom experiment name
        
    Returns:
        Complete DistillationConfig
    """
    
    datasets = get_dataset_configs()
    architectures = get_student_architectures()
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(datasets.keys())}")
    
    if student_arch not in architectures:
        raise ValueError(f"Architecture '{student_arch}' not found. Available: {list(architectures.keys())}")
    
    dataset_config = datasets[dataset_name]
    arch_config = architectures[student_arch]
    
    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = f"grounding_dino_to_{student_arch}_{dataset_name}"
    
    return DistillationConfig(
        # Dataset settings
        dataset_name=dataset_config["dataset_name"],
        num_classes=dataset_config["num_classes"],
        loader_function=dataset_config["loader_function"],
        train_split=dataset_config["train_split"],
        val_split=dataset_config["val_split"],
        batch_size=dataset_config["batch_size"],
        dataset_kwargs=dataset_config.get("dataset_kwargs"),
        
        # Student settings
        student_type=arch_config["student_type"],
        student_model=arch_config["student_model"],
        
        # Training settings
        num_epochs=num_epochs,
        learning_rate=1e-4,
        weight_decay=0.01,
        scheduler_type="cosine",
        
        # Distillation settings optimized for GroundingDINO
        temperature=3.5,  # Slightly lower for better grounding transfer
        alpha=0.8,        # Higher weight on features (grounding features are crucial)
        beta=0.2,         # Lower weight on attention
        gamma=1.0,        # Standard task loss
        
        # Checkpointing
        save_every=5,
        experiment_name=experiment_name
    )


def main():
    """
    Main experiment runner - CHANGE THESE LINES TO SWITCH EXPERIMENTS!
    """
    
    # ===== EXPERIMENT CONFIGURATION =====
    dataset_name = "oxford_pets"  # Change dataset here
    student_arch = "vit_base"     # Change student architecture here
    num_epochs = 20               # Adjust training length
    # ====================================
    
    print("🔍 GroundingDINO → Student Architecture Distillation")
    print("=" * 70)
    
    # Available options
    datasets = get_dataset_configs()
    architectures = get_student_architectures()
    
    print(f"📊 Available Datasets ({len(datasets)}):")
    for name, config in datasets.items():
        print(f"  📁 {name}: {config['description']}")
        print(f"     {config['grounding_benefit']}")
    
    print(f"\n🏗️ Available Student Architectures ({len(architectures)}):")
    for name, config in architectures.items():
        print(f"  🎯 {name}: {config['description']}")
        print(f"     {config['recommendation']} ({config['params']}, {config['speed']})")
    
    print("\n" + "=" * 70)
    print(f"🎯 SELECTED EXPERIMENT:")
    print(f"   Dataset: {datasets[dataset_name]['dataset_name']}")
    print(f"   Student: {architectures[student_arch]['student_type'].upper()} ({student_arch})")
    print(f"   Epochs: {num_epochs}")
    print(f"   Grounding Benefit: {datasets[dataset_name]['grounding_benefit']}")
    print(f"   Architecture Benefit: {architectures[student_arch]['recommendation']}")
    
    # Create configuration
    config = create_experiment_config(dataset_name, student_arch, num_epochs)
    
    # Create and run trainer
    print(f"\n🚀 Starting experiment...")
    trainer = GroundingDINODistillationTrainer(config)
    history = trainer.train()
    
    print(f"\n✅ Experiment completed!")
    print(f"📁 Results saved in: {trainer.exp_dir}")


def run_architecture_comparison():
    """
    Run comparison of different student architectures on the same dataset.
    """
    
    # Fixed dataset for comparison
    dataset_name = "oxford_pets"
    
    # Architectures to compare
    architectures_to_test = [
        "vit_base",      # Baseline
        "mae_base",      # MAE pretrained
        "deit_base",     # Distillation expert
        "swin_small",    # Hierarchical
        "convnext_small" # Modern ConvNet
    ]
    
    print("🔬 Student Architecture Comparison")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🏗️ Architectures: {architectures_to_test}")
    print("=" * 60)
    
    results = {}
    
    for arch in architectures_to_test:
        print(f"\n🎯 Testing {arch}...")
        
        try:
            # Create config with fewer epochs for comparison
            config = create_experiment_config(
                dataset_name=dataset_name,
                student_arch=arch,
                num_epochs=10,  # Shorter for comparison
                experiment_name=f"comparison_{arch}_{dataset_name}"
            )
            
            # Train
            trainer = GroundingDINODistillationTrainer(config)
            history = trainer.train()
            
            # Store results
            final_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
            results[arch] = {
                'accuracy': final_acc,
                'best_accuracy': max(history['val_accuracy']) if history['val_accuracy'] else 0,
                'experiment_dir': trainer.exp_dir
            }
            
            print(f"✅ {arch}: {final_acc:.2f}% accuracy")
            
        except Exception as e:
            print(f"❌ {arch}: Failed - {e}")
            results[arch] = {'accuracy': 0, 'best_accuracy': 0, 'error': str(e)}
    
    # Print comparison results
    print("\n" + "=" * 60)
    print("📊 ARCHITECTURE COMPARISON RESULTS")
    print("=" * 60)
    
    # Sort by best accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('best_accuracy', 0), reverse=True)
    
    for rank, (arch, result) in enumerate(sorted_results, 1):
        if 'error' not in result:
            print(f"{rank}. {arch:15} | Best: {result['best_accuracy']:.2f}% | Final: {result['accuracy']:.2f}%")
        else:
            print(f"{rank}. {arch:15} | FAILED: {result['error'][:40]}...")
    
    print("=" * 60)
    
    if sorted_results:
        winner = sorted_results[0]
        print(f"🏆 WINNER: {winner[0]} with {winner[1]['best_accuracy']:.2f}% accuracy")


def run_dataset_comparison():
    """
    Run comparison of different datasets with the same student architecture.
    """
    
    # Fixed student architecture
    student_arch = "vit_base"
    
    # Datasets to compare (smaller ones for speed)
    datasets_to_test = [
        "imagenette",    # High quality, 10 classes
        "oxford_pets",   # Medium classes, high quality
        "stl10",         # Medium quality
        "food101"        # Many classes, subset
    ]
    
    print("🔬 Dataset Comparison")
    print(f"🏗️ Student: {student_arch}")
    print(f"📊 Datasets: {datasets_to_test}")
    print("=" * 60)
    
    results = {}
    
    for dataset in datasets_to_test:
        print(f"\n📁 Testing {dataset}...")
        
        try:
            config = create_experiment_config(
                dataset_name=dataset,
                student_arch=student_arch,
                num_epochs=15,
                experiment_name=f"dataset_comparison_{dataset}_{student_arch}"
            )
            
            trainer = GroundingDINODistillationTrainer(config)
            history = trainer.train()
            
            final_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
            results[dataset] = {
                'accuracy': final_acc,
                'best_accuracy': max(history['val_accuracy']) if history['val_accuracy'] else 0,
                'num_classes': config.num_classes
            }
            
            print(f"✅ {dataset}: {final_acc:.2f}% accuracy")
            
        except Exception as e:
            print(f"❌ {dataset}: Failed - {e}")
            results[dataset] = {'accuracy': 0, 'best_accuracy': 0, 'error': str(e)}
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 DATASET COMPARISON RESULTS")
    print("=" * 60)
    
    for dataset, result in results.items():
        if 'error' not in result:
            print(f"{dataset:15} | {result['num_classes']:3d} classes | Best: {result['best_accuracy']:.2f}%")
        else:
            print(f"{dataset:15} | FAILED: {result['error'][:40]}...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "compare_arch":
            run_architecture_comparison()
        elif command == "compare_data":
            run_dataset_comparison()
        elif command == "main":
            main()
        else:
            print("Available commands:")
            print("  python student_architecture_experiments.py main")
            print("  python student_architecture_experiments.py compare_arch")
            print("  python student_architecture_experiments.py compare_data")
    else:
        # Default: run main experiment
        main()