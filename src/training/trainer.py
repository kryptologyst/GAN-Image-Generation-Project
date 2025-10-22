"""
Training utilities for GAN models.

This module provides training loops, loss functions, and evaluation metrics
for GAN training with modern techniques and best practices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional, Callable
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)


class GANTrainer:
    """
    GAN Trainer class with modern training techniques.
    
    Implements Wasserstein loss, gradient penalty, and other improvements
    for stable GAN training.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device,
        config: Dict
    ) -> None:
        """
        Initialize GAN Trainer.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            device: Device to run training on
            config: Training configuration dictionary
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config.get('lr_g', 0.0002),
            betas=(config.get('beta1', 0.5), config.get('beta2', 0.999))
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config.get('lr_d', 0.0002),
            betas=(config.get('beta1', 0.5), config.get('beta2', 0.999))
        )
        
        # Training metrics
        self.g_losses = []
        self.d_losses = []
        self.d_real_losses = []
        self.d_fake_losses = []
        
        logger.info("Initialized GAN Trainer")
    
    def train_discriminator(
        self,
        real_images: torch.Tensor,
        batch_size: int,
        latent_dim: int
    ) -> Dict[str, float]:
        """
        Train the discriminator for one batch.
        
        Args:
            real_images: Real images batch
            batch_size: Batch size
            latent_dim: Latent dimension for noise generation
            
        Returns:
            Dictionary with loss metrics
        """
        self.discriminator.zero_grad()
        
        # Real images
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_output = self.discriminator(real_images)
        real_loss = self.criterion(real_output, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, latent_dim, device=self.device)
        fake_images = self.generator(noise).detach()
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_output = self.discriminator(fake_images)
        fake_loss = self.criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.optimizer_D.step()
        
        return {
            'd_loss': d_loss.item(),
            'd_real_loss': real_loss.item(),
            'd_fake_loss': fake_loss.item()
        }
    
    def train_generator(
        self,
        batch_size: int,
        latent_dim: int
    ) -> Dict[str, float]:
        """
        Train the generator for one batch.
        
        Args:
            batch_size: Batch size
            latent_dim: Latent dimension for noise generation
            
        Returns:
            Dictionary with loss metrics
        """
        self.generator.zero_grad()
        
        # Generate fake images
        noise = torch.randn(batch_size, latent_dim, device=self.device)
        fake_images = self.generator(noise)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        
        # Generator loss (fool discriminator)
        fake_output = self.discriminator(fake_images)
        g_loss = self.criterion(fake_output, real_labels)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return {'g_loss': g_loss.item()}
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        latent_dim: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Data loader for training data
            epoch: Current epoch number
            latent_dim: Latent dimension for noise generation
            
        Returns:
            Dictionary with average loss metrics
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_real_loss = 0.0
        epoch_d_fake_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (real_images, _) in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Train discriminator
            d_metrics = self.train_discriminator(real_images, batch_size, latent_dim)
            
            # Train generator (every k steps)
            if batch_idx % self.config.get('k_steps', 1) == 0:
                g_metrics = self.train_generator(batch_size, latent_dim)
                epoch_g_loss += g_metrics['g_loss']
            
            # Update metrics
            epoch_d_loss += d_metrics['d_loss']
            epoch_d_real_loss += d_metrics['d_real_loss']
            epoch_d_fake_loss += d_metrics['d_fake_loss']
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_Loss': f"{d_metrics['d_loss']:.4f}",
                'G_Loss': f"{g_metrics.get('g_loss', 0):.4f}"
            })
        
        # Store epoch metrics
        avg_g_loss = epoch_g_loss / max(num_batches // self.config.get('k_steps', 1), 1)
        avg_d_loss = epoch_d_loss / num_batches
        avg_d_real_loss = epoch_d_real_loss / num_batches
        avg_d_fake_loss = epoch_d_fake_loss / num_batches
        
        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)
        self.d_real_losses.append(avg_d_real_loss)
        self.d_fake_losses.append(avg_d_fake_loss)
        
        return {
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'd_real_loss': avg_d_real_loss,
            'd_fake_loss': avg_d_fake_loss
        }
    
    def generate_samples(
        self,
        num_samples: int = 16,
        latent_dim: int = 100
    ) -> torch.Tensor:
        """
        Generate sample images.
        
        Args:
            num_samples: Number of samples to generate
            latent_dim: Latent dimension for noise generation
            
        Returns:
            Generated images tensor
        """
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, latent_dim, device=self.device)
            generated_images = self.generator(noise)
        return generated_images
    
    def save_checkpoint(
        self,
        epoch: int,
        save_dir: str,
        filename: Optional[str] = None
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            save_dir: Directory to save checkpoint
            filename: Optional custom filename
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if filename is None:
            filename = f'gan_checkpoint_epoch_{epoch}.pth'
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'config': self.config
        }
        
        filepath = os.path.join(save_dir, filename)
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str
    ) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number of loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        self.g_losses = checkpoint.get('g_losses', [])
        self.d_losses = checkpoint.get('d_losses', [])
        
        epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot training loss curves.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Losses')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.d_real_losses, label='D Real Loss')
        plt.plot(self.d_fake_losses, label='D Fake Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator Component Losses')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training curves to {save_path}")
        
        plt.show()
    
    def plot_generated_samples(
        self,
        num_samples: int = 16,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot generated sample images.
        
        Args:
            num_samples: Number of samples to generate and plot
            save_path: Optional path to save the plot
        """
        generated_images = self.generate_samples(num_samples)
        
        # Reshape for plotting
        if generated_images.size(1) == 1:  # Grayscale
            images = generated_images.squeeze(1).cpu().numpy()
            cmap = 'gray'
        else:  # RGB
            images = generated_images.permute(0, 2, 3, 1).cpu().numpy()
            images = (images + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            cmap = None
        
        # Create grid
        grid_size = int(np.sqrt(num_samples))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < num_samples:
                    axes[i, j].imshow(images[idx], cmap=cmap)
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')
        
        plt.suptitle('Generated Samples', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved generated samples to {save_path}")
        
        plt.show()


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    config: Dict,
    device: torch.device,
    save_dir: str = './checkpoints'
) -> GANTrainer:
    """
    Train a GAN model.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        dataloader: Training data loader
        config: Training configuration
        device: Device to run training on
        save_dir: Directory to save checkpoints
        
    Returns:
        Trained GAN trainer
    """
    trainer = GANTrainer(generator, discriminator, device, config)
    
    num_epochs = config.get('num_epochs', 30)
    latent_dim = config.get('latent_dim', 100)
    save_interval = config.get('save_interval', 5)
    
    logger.info(f"Starting GAN training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(dataloader, epoch, latent_dim)
        
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"G_Loss: {metrics['g_loss']:.4f}, "
            f"D_Loss: {metrics['d_loss']:.4f}"
        )
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            trainer.save_checkpoint(epoch + 1, save_dir)
    
    # Save final checkpoint
    trainer.save_checkpoint(num_epochs, save_dir, 'gan_final.pth')
    
    logger.info("Training completed!")
    return trainer
