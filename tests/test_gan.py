"""
Unit tests for GAN models and training utilities.

This module contains comprehensive tests for the GAN implementation,
including model creation, training, and utility functions.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.dcgan import (
    DCGANGenerator, DCGANDiscriminator,
    SimpleGANGenerator, SimpleGANDiscriminator,
    get_model, initialize_weights
)
from src.training.trainer import GANTrainer
from src.utils.data_utils import (
    SyntheticDataset, get_mnist_dataloader,
    calculate_fid_score, save_generated_images,
    load_config, save_config
)


class TestDCGANModels(unittest.TestCase):
    """Test DCGAN model implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.latent_dim = 100
        self.image_size = 64
        self.num_channels = 3
        self.feature_map_size = 64
        self.batch_size = 4
    
    def test_dcgan_generator_creation(self):
        """Test DCGAN generator creation."""
        generator = DCGANGenerator(
            latent_dim=self.latent_dim,
            num_channels=self.num_channels,
            feature_map_size=self.feature_map_size,
            image_size=self.image_size
        )
        
        self.assertIsInstance(generator, DCGANGenerator)
        self.assertEqual(generator.latent_dim, self.latent_dim)
        self.assertEqual(generator.num_channels, self.num_channels)
        self.assertEqual(generator.image_size, self.image_size)
    
    def test_dcgan_generator_forward(self):
        """Test DCGAN generator forward pass."""
        generator = DCGANGenerator(
            latent_dim=self.latent_dim,
            num_channels=self.num_channels,
            image_size=self.image_size
        )
        
        # Test forward pass
        noise = torch.randn(self.batch_size, self.latent_dim)
        output = generator(noise)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_channels, self.image_size, self.image_size)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output range (should be in [-1, 1] due to Tanh)
        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))
    
    def test_dcgan_discriminator_creation(self):
        """Test DCGAN discriminator creation."""
        discriminator = DCGANDiscriminator(
            num_channels=self.num_channels,
            feature_map_size=self.feature_map_size,
            image_size=self.image_size
        )
        
        self.assertIsInstance(discriminator, DCGANDiscriminator)
        self.assertEqual(discriminator.num_channels, self.num_channels)
        self.assertEqual(discriminator.image_size, self.image_size)
    
    def test_dcgan_discriminator_forward(self):
        """Test DCGAN discriminator forward pass."""
        discriminator = DCGANDiscriminator(
            num_channels=self.num_channels,
            image_size=self.image_size
        )
        
        # Test forward pass
        images = torch.randn(self.batch_size, self.num_channels, self.image_size, self.image_size)
        output = discriminator(images)
        
        # Check output shape
        expected_shape = (self.batch_size, 1)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output range (should be in [0, 1] due to Sigmoid)
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))


class TestSimpleGANModels(unittest.TestCase):
    """Test Simple GAN model implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.latent_dim = 100
        self.image_size = 28
        self.num_channels = 1
        self.batch_size = 4
    
    def test_simple_gan_generator_creation(self):
        """Test Simple GAN generator creation."""
        generator = SimpleGANGenerator(
            latent_dim=self.latent_dim,
            image_size=self.image_size,
            num_channels=self.num_channels
        )
        
        self.assertIsInstance(generator, SimpleGANGenerator)
        self.assertEqual(generator.latent_dim, self.latent_dim)
        self.assertEqual(generator.image_size, self.image_size)
        self.assertEqual(generator.num_channels, self.num_channels)
    
    def test_simple_gan_generator_forward(self):
        """Test Simple GAN generator forward pass."""
        generator = SimpleGANGenerator(
            latent_dim=self.latent_dim,
            image_size=self.image_size,
            num_channels=self.num_channels
        )
        
        # Test forward pass
        noise = torch.randn(self.batch_size, self.latent_dim)
        output = generator(noise)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_channels, self.image_size, self.image_size)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output range (should be in [-1, 1] due to Tanh)
        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))
    
    def test_simple_gan_discriminator_creation(self):
        """Test Simple GAN discriminator creation."""
        discriminator = SimpleGANDiscriminator(
            image_size=self.image_size,
            num_channels=self.num_channels
        )
        
        self.assertIsInstance(discriminator, SimpleGANDiscriminator)
        self.assertEqual(discriminator.image_size, self.image_size)
        self.assertEqual(discriminator.num_channels, self.num_channels)
    
    def test_simple_gan_discriminator_forward(self):
        """Test Simple GAN discriminator forward pass."""
        discriminator = SimpleGANDiscriminator(
            image_size=self.image_size,
            num_channels=self.num_channels
        )
        
        # Test forward pass
        images = torch.randn(self.batch_size, self.num_channels, self.image_size, self.image_size)
        output = discriminator(images)
        
        # Check output shape
        expected_shape = (self.batch_size, 1)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output range (should be in [0, 1] due to Sigmoid)
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))


class TestModelFactory(unittest.TestCase):
    """Test model factory function."""
    
    def test_get_model_simple(self):
        """Test getting simple GAN models."""
        generator, discriminator = get_model(
            model_type="simple",
            latent_dim=100,
            image_size=28,
            num_channels=1
        )
        
        self.assertIsInstance(generator, SimpleGANGenerator)
        self.assertIsInstance(discriminator, SimpleGANDiscriminator)
    
    def test_get_model_dcgan(self):
        """Test getting DCGAN models."""
        generator, discriminator = get_model(
            model_type="dcgan",
            latent_dim=100,
            image_size=64,
            num_channels=3
        )
        
        self.assertIsInstance(generator, DCGANGenerator)
        self.assertIsInstance(discriminator, DCGANDiscriminator)


class TestSyntheticDataset(unittest.TestCase):
    """Test synthetic dataset implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_samples = 100
        self.image_size = 28
        self.num_channels = 1
    
    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticDataset(
            num_samples=self.num_samples,
            image_size=self.image_size,
            num_channels=self.num_channels,
            pattern_type="circles"
        )
        
        self.assertEqual(len(dataset), self.num_samples)
        self.assertEqual(dataset.image_size, self.image_size)
        self.assertEqual(dataset.num_channels, self.num_channels)
    
    def test_synthetic_dataset_getitem(self):
        """Test synthetic dataset item retrieval."""
        dataset = SyntheticDataset(
            num_samples=self.num_samples,
            image_size=self.image_size,
            pattern_type="circles"
        )
        
        image, label = dataset[0]
        
        # Check image shape
        expected_shape = (self.num_channels, self.image_size, self.image_size)
        self.assertEqual(image.shape, expected_shape)
        
        # Check image range (should be in [-1, 1])
        self.assertTrue(torch.all(image >= -1.0))
        self.assertTrue(torch.all(image <= 1.0))
        
        # Check label
        self.assertEqual(label, 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_calculate_fid_score(self):
        """Test FID score calculation."""
        # Create dummy features
        real_features = np.random.randn(100, 10)
        fake_features = np.random.randn(100, 10)
        
        fid_score = calculate_fid_score(real_features, fake_features)
        
        self.assertIsInstance(fid_score, float)
        self.assertGreaterEqual(fid_score, 0.0)
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        config = {
            "model": {"type": "simple", "latent_dim": 100},
            "training": {"epochs": 30, "batch_size": 128}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save config
            save_config(config, temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load config
            loaded_config = load_config(temp_path)
            self.assertEqual(loaded_config, config)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_generated_images(self):
        """Test saving generated images."""
        # Create dummy images
        images = torch.randn(4, 1, 28, 28)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_generated_images(images, temp_dir, prefix="test")
            
            # Check if images were saved
            saved_files = os.listdir(temp_dir)
            self.assertEqual(len(saved_files), 4)
            
            for i in range(4):
                expected_filename = f"test_{i:04d}.png"
                self.assertIn(expected_filename, saved_files)


class TestGANTrainer(unittest.TestCase):
    """Test GAN trainer implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.config = {
            "lr_g": 0.0002,
            "lr_d": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999,
            "k_steps": 1
        }
        
        # Create simple models
        self.generator = SimpleGANGenerator(latent_dim=100, image_size=28)
        self.discriminator = SimpleGANDiscriminator(image_size=28)
    
    def test_trainer_initialization(self):
        """Test GAN trainer initialization."""
        trainer = GANTrainer(
            generator=self.generator,
            discriminator=self.discriminator,
            device=self.device,
            config=self.config
        )
        
        self.assertIsInstance(trainer, GANTrainer)
        self.assertEqual(trainer.device, self.device)
        self.assertEqual(trainer.config, self.config)
    
    def test_generate_samples(self):
        """Test sample generation."""
        trainer = GANTrainer(
            generator=self.generator,
            discriminator=self.discriminator,
            device=self.device,
            config=self.config
        )
        
        samples = trainer.generate_samples(num_samples=4, latent_dim=100)
        
        # Check output shape
        expected_shape = (4, 1, 28, 28)
        self.assertEqual(samples.shape, expected_shape)
        
        # Check output range
        self.assertTrue(torch.all(samples >= -1.0))
        self.assertTrue(torch.all(samples <= 1.0))


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
