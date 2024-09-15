import torch
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset, Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

"""# Dataset and dataloader"""

# Hyperparameters and paths
num_epochs = 500
batch_size = 64
warmup_epochs = 75
best_psnr = 0
beta = 1.0
checkpoint_path = "/working/"    #update it with your checkpoints path 

# Paths to dataset directories
PATHS = [
    "imagenet100/train.X1",
    "imagenet100/train.X2",
    "imagenet100/train.X3",
    "imagenet100/train.X4"
]

# Transformation pipeline for training
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.RandomCrop(64),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformation pipeline for validation and testing
val_test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformation pipeline for visualisation of validation and testing
visual_val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets from multiple folders and combine them
train_val_test_datasets = [datasets.ImageFolder(path) for path in PATHS]
train_val_test_dataset = ConcatDataset(train_val_test_datasets)

def split_dataset(dataset, sizes):
    """Split the dataset into specified sizes."""
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    split_indices = []
    start_idx = 0
    for size in sizes:
        split_indices.append(indices[start_idx:start_idx + size])
        start_idx += size

    return split_indices

# Specify sizes for the splits
split_sizes = [15000, 2000, 1000, 1000]
split_indices = split_dataset(train_val_test_dataset, split_sizes)

# Create subsets for the splits
train_dataset = Subset(train_val_test_dataset, split_indices[0])
val_dataset = Subset(train_val_test_dataset, split_indices[1])
test_dataset = Subset(train_val_test_dataset, split_indices[2])
visual_dataset = Subset(train_val_test_dataset, split_indices[3])

# Custom dataset class to apply different transforms
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x), y

    def __len__(self):
        return len(self.subset)

# Apply appropriate transforms to the datasets
train_dataset = TransformDataset(train_dataset, train_transform)
val_dataset = TransformDataset(val_dataset, val_test_transform)
test_dataset = TransformDataset(test_dataset, val_test_transform)
visual_dataset = TransformDataset(visual_dataset, visual_val_test_transform)

# Create DataLoaders for all datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
visual_loader = DataLoader(visual_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

def show_images(dataset, indices):
    """Visualize images from the dataset at specified indices."""
    plt.figure(figsize=(15, 15))

    # Display images
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image = image.permute(1, 2, 0)  
        image = image.numpy()

        # Denormalize the image for proper visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        plt.subplot(5, 10, i + 1)  
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Label: {label}')

    plt.show()

if __name__ == "__main__":
    show_images(train_dataset, indices=range(50))
