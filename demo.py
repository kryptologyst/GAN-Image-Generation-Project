#!/usr/bin/env python3
"""
Demo script for GAN Image Generation.

This script demonstrates the key features of the GAN implementation
with a simple example that can be run quickly.
"""

import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.dcgan import get_model
from src.utils.data_utils import get_synthetic_dataloader, setup_logging


def main():
    """Run a quick demo of the GAN implementation."""
    print("🎨 GAN Image Generation Demo")
    print("=" * 40)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models
    print("\n📦 Creating GAN models...")
    generator, discriminator = get_model(
        model_type="simple",
        latent_dim=100,
        image_size=28,
        num_channels=1
    )
    
    generator.to(device)
    discriminator.to(device)
    
    print(f"✅ Created {type(generator).__name__} and {type(discriminator).__name__}")
    
    # Create synthetic dataset
    print("\n📊 Creating synthetic dataset...")
    dataloader = get_synthetic_dataloader(
        batch_size=16,
        num_samples=100,
        image_size=28,
        pattern_type="circles"
    )
    
    print(f"✅ Created dataset with {len(dataloader)} batches")
    
    # Test forward passes
    print("\n🧪 Testing model forward passes...")
    
    # Test generator
    noise = torch.randn(4, 100, device=device)
    generated_images = generator(noise)
    print(f"✅ Generator output shape: {generated_images.shape}")
    
    # Test discriminator
    real_images = torch.randn(4, 1, 28, 28, device=device)
    discriminator_output = discriminator(real_images)
    print(f"✅ Discriminator output shape: {discriminator_output.shape}")
    
    # Generate and visualize samples
    print("\n🎨 Generating sample images...")
    generator.eval()
    with torch.no_grad():
        sample_noise = torch.randn(16, 100, device=device)
        sample_images = generator(sample_noise)
    
    # Convert to numpy for visualization
    sample_images_np = sample_images.cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        row, col = i // 4, i % 4
        axes[row, col].imshow(sample_images_np[i, 0], cmap='gray')
        axes[row, col].axis('off')
    
    plt.suptitle('Generated Samples (Untrained)', fontsize=16)
    plt.tight_layout()
    plt.savefig('demo_samples.png', dpi=150, bbox_inches='tight')
    print("✅ Saved generated samples to 'demo_samples.png'")
    
    # Show dataset samples
    print("\n📈 Visualizing dataset samples...")
    data_iter = iter(dataloader)
    real_images, _ = next(data_iter)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        row, col = i // 4, i % 4
        img = real_images[i, 0].numpy()  # This should be (28, 28)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
    
    plt.suptitle('Real Dataset Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('demo_dataset.png', dpi=150, bbox_inches='tight')
    print("✅ Saved dataset samples to 'demo_dataset.png'")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python train.py --model-type simple --dataset mnist --epochs 10' to train a model")
    print("2. Run 'streamlit run web_app/app.py' to launch the web interface")
    print("3. Run 'python -m pytest tests/' to run the test suite")


if __name__ == "__main__":
    main()
