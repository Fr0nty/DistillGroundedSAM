# install_requirements.py
"""
Simple installation script for GroundingDINO Distillation dependencies.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def main():
    """Install all required packages."""
    
    print("Installing GroundingDINO Distillation Dependencies")
    print("=" * 50)
    
    # Required packages
    packages = [
        "torch>=1.12.0",
        "torchvision>=0.13.0", 
        "timm>=0.6.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "thop",  # For FLOPs calculation
        "ptflops",  # Alternative FLOPs calculation
        "tqdm",
        "numpy",
        "Pillow",
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nInstallation Summary: {success_count}/{total_count} packages installed")
    
    if success_count == total_count:
        print("All dependencies installed successfully!")
        print("\nOptional: Install GroundingDINO for best results:")
        print("  pip install groundingdino-py")
        print("  OR")
        print("  git clone https://github.com/IDEA-Research/GroundingDINO.git")
        print("  cd GroundingDINO && pip install -e .")
        
        print("\nNext steps:")
        print("  1. Run training: python student_architecture_experiments.py")
        print("  2. Run evaluation: python evaluation_system.py multi")
    else:
        print("Some packages failed to install. Check errors above.")

if __name__ == "__main__":
    main()