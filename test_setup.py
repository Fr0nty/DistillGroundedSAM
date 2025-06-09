#!/usr/bin/env python3
# test_setup.py - Test CUDA fixes and module imports

import torch
import sys
from pathlib import Path

def test_cuda_setup():
    """Test CUDA setup."""
    print("Testing CUDA Setup")
    print("=" * 30)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        
        # Test basic operations
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            memory_used = torch.cuda.memory_allocated() // 1024**2
            print(f"Basic CUDA operations work - {memory_used}MB GPU memory used")
            del x, y
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"CUDA operations failed: {e}")
            return False
    else:
        print("CUDA not available - will use CPU")
    
    return True

def test_imports():
    """Test required imports."""
    print("\nTesting Imports")
    print("=" * 30)
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("timm", "TIMM"),
        ("transformers", "Transformers"),
        ("datasets", "HF Datasets"),
    ]
    
    success_count = 0
    
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"✓ {name}")
            success_count += 1
        except ImportError:
            print(f"✗ {name} - install with: pip install {module}")
    
    print(f"\nImport Summary: {success_count}/{len(imports_to_test)} packages available")
    return success_count == len(imports_to_test)

def test_project_structure():
    """Test project file structure."""
    print("\nTesting Project Structure")
    print("=" * 30)
    
    required_files = [
        "GrdDINO_Distill.py",
        "student_architecture_experiments.py",
        "data/multiple_datasets.py",
        "evaluation_system.py"
    ]
    
    found_count = 0
    
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
            found_count += 1
        else:
            print(f"✗ {file} - missing")
    
    print(f"\nFile Summary: {found_count}/{len(required_files)} files found")
    return found_count == len(required_files)

def test_cuda_fixes():
    """Test CUDA fixes by creating a simple teacher model."""
    print("\nTesting CUDA Fixes")
    print("=" * 30)
    
    try:
        # Try importing the main classes
        from GrdDINO_Distill import GroundingDINOTeacher, MultiStudentArchitecture
        print("✓ Main classes imported successfully")
        
        # Test teacher initialization with CUDA safety
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        teacher = GroundingDINOTeacher(device=device)
        print(f"✓ Teacher initialized on {teacher.device}")
        
        # Test student initialization
        student = MultiStudentArchitecture(
            student_type="vit",
            model_name="vit_small_patch16_224",  # Use smaller model for testing
            num_classes=10
        )
        student = student.to(torch.device(device))
        print(f"✓ Student initialized")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(torch.device(device))
        
        with torch.no_grad():
            teacher_output = teacher(dummy_input)
            student_output = student(dummy_input)
        
        print("✓ Forward pass successful")
        print(f"  Teacher output keys: {list(teacher_output.keys())}")
        print(f"  Student output keys: {list(student_output.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ CUDA fixes test failed: {e}")
        return False

def test_evaluation_system():
    """Test evaluation system."""
    print("\nTesting Evaluation System")
    print("=" * 30)
    
    try:
        # Check if experiments directory exists
        exp_dir = Path("experiments")
        if exp_dir.exists():
            subdirs = [d for d in exp_dir.iterdir() if d.is_dir()]
            print(f"✓ Found {len(subdirs)} experiment directories")
            for subdir in subdirs[:3]:  # Show first 3
                print(f"  - {subdir.name}")
        else:
            print("! No experiments directory found - run training first")
        
        # Test evaluation imports
        from evaluation_system import ModelMetrics, DistillationEvaluator, EvaluationPlotter
        print("✓ Evaluation system imports work")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation system test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("GroundingDINO Distillation Test Suite")
    print("=" * 60)
    
    tests = [
        ("CUDA Setup", test_cuda_setup),
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("CUDA Fixes", test_cuda_fixes),
        ("Evaluation System", test_evaluation_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\nTEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} | {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\nAll tests passed! You're ready to run distillation.")
        print("Next steps:")
        print("  1. python student_architecture_experiments.py")
        print("  2. python evaluation_system.py multi")
    elif passed >= len(results) - 1:
        print("\nMost tests passed. You can try running the distillation.")
        print("Some non-critical features may not work.")
    else:
        print("\nSeveral tests failed. Please check the errors above.")
        print("Run: python install_requirements.py")

if __name__ == "__main__":
    main()