import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class HuggingFaceCIFARDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Only convert to PIL.Image if not already
        img = sample["img"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = sample["label"]
        return img, label

def get_cifar_dataloader(
    split: str = "train[:10%]",
    dataset_name: str = "cifar10",
    batch_size: int = 32,
    img_size: int = 224,
    shuffle: bool = True,
    num_workers: int = 4,
):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    # Load subset of the dataset
    hf_dataset = load_dataset(dataset_name, split=split)
    dataset = HuggingFaceCIFARDataset(hf_dataset, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return loader
