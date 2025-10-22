"""
Utility functions for data handling, visualization, and model evaluation.

This module provides helper functions for data preprocessing, visualization,
and evaluation metrics for GAN models.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any
import logging
import os
from PIL import Image
import json

logger = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
    """
    Synthetic dataset generator for testing and demonstration.
    
    Creates synthetic images with simple patterns for GAN training
    when real datasets are not available.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 28,
        num_channels: int = 1,
        pattern_type: str = "circles"
    ) -> None:
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Size of generated images (assumed square)
            num_channels: Number of channels in images
            pattern_type: Type of pattern to generate ("circles", "squares", "lines")
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        self.pattern_type = pattern_type
        
        logger.info(f"Created synthetic dataset with {num_samples} samples, "
                   f"pattern: {pattern_type}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Generate a synthetic image.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        if self.pattern_type == "circles":
            image = self._generate_circle_pattern()
        elif self.pattern_type == "squares":
            image = self._generate_square_pattern()
        elif self.pattern_type == "lines":
            image = self._generate_line_pattern()
        else:
            image = self._generate_random_pattern()
        
        # Convert to tensor and normalize to [-1, 1]
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # Reshape to (channels, height, width)
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
        
        return image_tensor, 0  # Dummy label
    
    def _generate_circle_pattern(self) -> np.ndarray:
        """Generate a circle pattern."""
        image = np.zeros((self.image_size, self.image_size))
        
        # Random circle parameters
        center_x = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
        center_y = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
        radius = np.random.randint(3, self.image_size // 4)
        
        # Create circle
        y, x = np.ogrid[:self.image_size, :self.image_size]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        image[mask] = 1.0
        
        return image
    
    def _generate_square_pattern(self) -> np.ndarray:
        """Generate a square pattern."""
        image = np.zeros((self.image_size, self.image_size))
        
        # Random square parameters
        size = np.random.randint(5, self.image_size // 2)
        start_x = np.random.randint(0, self.image_size - size)
        start_y = np.random.randint(0, self.image_size - size)
        
        # Create square
        image[start_y:start_y+size, start_x:start_x+size] = 1.0
        
        return image
    
    def _generate_line_pattern(self) -> np.ndarray:
        """Generate a line pattern."""
        image = np.zeros((self.image_size, self.image_size))
        
        # Random line parameters
        if np.random.random() > 0.5:  # Horizontal line
            y = np.random.randint(0, self.image_size)
            thickness = np.random.randint(1, 4)
            image[max(0, y-thickness//2):min(self.image_size, y+thickness//2+1), :] = 1.0
        else:  # Vertical line
            x = np.random.randint(0, self.image_size)
            thickness = np.random.randint(1, 4)
            image[:, max(0, x-thickness//2):min(self.image_size, x+thickness//2+1)] = 1.0
        
        return image
    
    def _generate_random_pattern(self) -> np.ndarray:
        """Generate a random noise pattern."""
        return np.random.random((self.image_size, self.image_size))


def get_mnist_dataloader(
    batch_size: int = 128,
    image_size: int = 28,
    download: bool = True,
    data_dir: str = './data'
) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST dataloaders for training and testing.
    
    Args:
        batch_size: Batch size for data loaders
        image_size: Size to resize images to
        download: Whether to download MNIST if not present
        data_dir: Directory to store MNIST data
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Transform: convert to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=download
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=download
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    logger.info(f"Created MNIST dataloaders with batch_size={batch_size}")
    return train_loader, test_loader


def get_cifar10_dataloader(
    batch_size: int = 128,
    image_size: int = 64,
    download: bool = True,
    data_dir: str = './data'
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 dataloaders for training and testing.
    
    Args:
        batch_size: Batch size for data loaders
        image_size: Size to resize images to
        download: Whether to download CIFAR-10 if not present
        data_dir: Directory to store CIFAR-10 data
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Transform: convert to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, transform=transform, download=download
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, transform=transform, download=download
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    logger.info(f"Created CIFAR-10 dataloaders with batch_size={batch_size}")
    return train_loader, test_loader


def get_synthetic_dataloader(
    batch_size: int = 128,
    num_samples: int = 1000,
    image_size: int = 28,
    pattern_type: str = "circles"
) -> DataLoader:
    """
    Get synthetic dataloader for testing.
    
    Args:
        batch_size: Batch size for data loader
        num_samples: Number of synthetic samples to generate
        image_size: Size of generated images
        pattern_type: Type of pattern to generate
        
    Returns:
        DataLoader with synthetic data
    """
    dataset = SyntheticDataset(num_samples, image_size, pattern_type=pattern_type)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Created synthetic dataloader with {num_samples} samples")
    return loader


def visualize_dataset_samples(
    dataloader: DataLoader,
    num_samples: int = 16,
    title: str = "Dataset Samples",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize samples from a dataset.
    
    Args:
        dataloader: DataLoader to visualize samples from
        num_samples: Number of samples to display
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    # Get a batch of samples
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Take only the requested number of samples
    images = images[:num_samples]
    
    # Reshape for plotting
    if images.size(1) == 1:  # Grayscale
        images_np = images.squeeze(1).numpy()
        cmap = 'gray'
    else:  # RGB
        images_np = images.permute(0, 2, 3, 1).numpy()
        images_np = (images_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        cmap = None
    
    # Create grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                axes[i, j].imshow(images_np[idx], cmap=cmap)
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dataset visualization to {save_path}")
    
    plt.show()


def calculate_fid_score(
    real_features: np.ndarray,
    fake_features: np.ndarray
) -> float:
    """
    Calculate Fréchet Inception Distance (FID) score.
    
    Args:
        real_features: Features from real images
        fake_features: Features from generated images
        
    Returns:
        FID score
    """
    # Calculate means and covariances
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean, _ = np.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % 1e-6
        logger.warning(msg)
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = np.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid


def save_generated_images(
    images: torch.Tensor,
    save_dir: str,
    prefix: str = "generated",
    denormalize: bool = True
) -> None:
    """
    Save generated images to disk.
    
    Args:
        images: Tensor of generated images
        save_dir: Directory to save images
        prefix: Prefix for saved filenames
        denormalize: Whether to denormalize images from [-1, 1] to [0, 1]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy and denormalize if needed
    if denormalize:
        images_np = (images.cpu().numpy() + 1) / 2
        images_np = np.clip(images_np, 0, 1)
    else:
        images_np = images.cpu().numpy()
    
    # Save each image
    for i, img in enumerate(images_np):
        if img.shape[0] == 1:  # Grayscale
            img = img.squeeze(0)
            img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
        else:  # RGB
            img = img.transpose(1, 2, 0)
            img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
        
        filename = f"{prefix}_{i:04d}.png"
        filepath = os.path.join(save_dir, filename)
        img_pil.save(filepath)
    
    logger.info(f"Saved {len(images)} images to {save_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging setup complete - Level: {log_level}")
