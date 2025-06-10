import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class Food101Dataset(Dataset):
    """Food101 Dataset - naturally high resolution images (512x512 typical)"""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = sample["label"]
        return img, label

class STL10Dataset(Dataset):
    """STL10 Dataset - 96x96 images (much less distortion than CIFAR)"""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = Image.fromarray(np.array(sample["image"]))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = sample["label"]
        return img, label

class OxfordPetsDataset(Dataset):
    """Oxford-IIIT Pet Dataset - variable high resolution images"""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = sample["label"]
        return img, label

class Flowers102Dataset(Dataset):
    """Oxford 102 Flower Dataset - high resolution flower images"""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = sample["label"]
        return img, label

class ImageNetteDataset(Dataset):
    """Imagenette - subset of ImageNet with 10 classes, high resolution"""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = sample["label"]
        return img, label

def get_food101_dataloader(
    split: str = "train[:10%]",
    batch_size: int = 32,
    img_size: int = 224,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """
    Food101 dataset - 101 food categories, naturally high resolution images
    Original images are typically 512x512, so minimal distortion when resized to 224x224
    
    Splits available: 'train', 'validation'
    Example splits: 'train[:10%]', 'validation[:50%]'
    """
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])

    print(f"Loading Food101 dataset: {split}")
    hf_dataset = load_dataset("food101", split=split)
    dataset = Food101Dataset(hf_dataset, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Food101 dataset loaded: {len(dataset)} samples, {len(loader)} batches")
    return loader

def get_stl10_dataloader(
    split: str = "train",
    batch_size: int = 32,
    img_size: int = 224,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """
    STL10 dataset - 10 classes, 96x96 images (much better than CIFAR's 32x32)
    Less distortion when resized to 224x224
    
    Splits available: 'train', 'test', 'unlabeled'
    """
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading STL10 dataset: {split}")
    hf_dataset = load_dataset("stl10", split=split)
    dataset = STL10Dataset(hf_dataset, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"STL10 dataset loaded: {len(dataset)} samples, {len(loader)} batches")
    return loader

def get_oxford_pets_dataloader(
    split: str = "train",
    batch_size: int = 32,
    img_size: int = 224,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """
    Oxford-IIIT Pet Dataset - 37 pet categories, variable high resolution images
    Original images are typically much larger than 224x224
    
    Splits available: 'train', 'test'
    """
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading Oxford Pets dataset: {split}")
    hf_dataset = load_dataset("timm/oxford-iiit-pet", split=split)
    dataset = OxfordPetsDataset(hf_dataset, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Oxford Pets dataset loaded: {len(dataset)} samples, {len(loader)} batches")
    return loader

def get_flowers102_dataloader(
    split: str = "train",
    batch_size: int = 32,
    img_size: int = 224,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """
    Oxford 102 Flower Dataset - 102 flower categories, high resolution images
    Original images are variable size but typically much larger than 224x224
    
    Splits available: 'train', 'validation', 'test'
    """
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading Flowers102 dataset: {split}")
    hf_dataset = load_dataset("nelorth/oxford-flowers", split=split)
    dataset = Flowers102Dataset(hf_dataset, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Flowers102 dataset loaded: {len(dataset)} samples, {len(loader)} batches")
    return loader

def get_imagenette_dataloader(
    split: str = "train",
    batch_size: int = 32,
    img_size: int = 224,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """
    Imagenette - subset of ImageNet with 10 classes, high resolution
    Original ImageNet images, so naturally high quality for 224x224
    
    Splits available: 'train', 'validation'
    """
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading Imagenette dataset: {split}")
    hf_dataset = load_dataset("frgfm/imagenette", "320px", split=split)
    dataset = ImageNetteDataset(hf_dataset, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Imagenette dataset loaded: {len(dataset)} samples, {len(loader)} batches")
    return loader

def compare_dataset_resolutions():
    """Compare original resolutions of different datasets"""
    
    datasets_info = {
        "CIFAR-10": {
            "original_size": "32x32",
            "distortion_factor": 224/32,
            "quality": "Poor (7x upscaling)"
        },
        "STL-10": {
            "original_size": "96x96", 
            "distortion_factor": 224/96,
            "quality": "Good (2.3x upscaling)"
        },
        "Food101": {
            "original_size": "~512x512",
            "distortion_factor": 224/512,
            "quality": "Excellent (downscaling)"
        },
        "Oxford Pets": {
            "original_size": "Variable (typically >300x300)",
            "distortion_factor": "~0.7x",
            "quality": "Excellent (minimal scaling)"
        },
        "Flowers102": {
            "original_size": "Variable (typically >400x400)",
            "distortion_factor": "~0.6x",
            "quality": "Excellent (downscaling)"
        },
        "Imagenette": {
            "original_size": "320x320",
            "distortion_factor": 224/320,
            "quality": "Very Good (0.7x downscaling)"
        }
    }
    
    print("Dataset Resolution Comparison for 224x224 training:")
    print("="*60)
    for name, info in datasets_info.items():
        print(f"{name:15} | {info['original_size']:20} | {info['quality']}")
    print("="*60)
    
    return datasets_info

# Example usage and training integration
def main():
    """Example of how to use the improved datasets"""
    
    # Compare resolutions
    compare_dataset_resolutions()
    
    print("\n" + "="*50)
    print("RECOMMENDED DATASETS FOR DINOV2 DISTILLATION:")
    print("="*50)
    
    # Try different datasets
    datasets_to_try = [
        ("Food101", get_food101_dataloader, "train[:5%]"),
        ("STL10", get_stl10_dataloader, "train"),
        ("Oxford Pets", get_oxford_pets_dataloader, "train"),
        ("Flowers102", get_flowers102_dataloader, "train"),
        ("Imagenette", get_imagenette_dataloader, "train"),
    ]
    
    for name, loader_func, split in datasets_to_try:
        try:
            print(f"\nTesting {name}...")
            loader = loader_func(split=split, batch_size=8, num_workers=2)
            
            # Test loading a batch
            batch = next(iter(loader))
            images, labels = batch
            
            print(f"✅ {name}: Batch shape {images.shape}, Labels shape {labels.shape}")
            print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"   Unique labels: {len(torch.unique(labels))}")
            
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    print("\n" + "="*50)
    print("RECOMMENDATION:")
    print("1. Food101: Best for food/object recognition")
    print("2. Oxford Pets: Good for animal classification") 
    print("3. STL10: Good balance, similar to CIFAR but higher res")
    print("4. Imagenette: Subset of ImageNet, very high quality")
    print("="*50)

if __name__ == "__main__":
    main()