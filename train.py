#!/usr/bin/env python3
"""
Main training script for GAN image generation.

This script provides a command-line interface for training GAN models
with various configurations and datasets.
"""

import argparse
import json
import logging
import os
import sys
import torch
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.dcgan import get_model
from src.training.trainer import train_gan
from src.utils.data_utils import (
    get_mnist_dataloader,
    get_cifar10_dataloader,
    get_synthetic_dataloader,
    setup_logging,
    load_config,
    save_config
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GAN for image generation")
    
    # Model arguments
    parser.add_argument("--model-type", type=str, default="simple",
                       choices=["simple", "dcgan"],
                       help="Type of GAN model to use")
    parser.add_argument("--latent-dim", type=int, default=100,
                       help="Dimension of latent noise vector")
    parser.add_argument("--image-size", type=int, default=28,
                       help="Size of generated images")
    parser.add_argument("--num-channels", type=int, default=1,
                       help="Number of channels in images")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for training")
    parser.add_argument("--lr-g", type=float, default=0.0002,
                       help="Learning rate for generator")
    parser.add_argument("--lr-d", type=float, default=0.0002,
                       help="Learning rate for discriminator")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "cifar10", "synthetic"],
                       help="Dataset to use for training")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Directory to store dataset")
    parser.add_argument("--synthetic-samples", type=int, default=1000,
                       help="Number of synthetic samples to generate")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Directory to save outputs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Logging arguments
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Path to log file")
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> dict:
    """Create configuration dictionary from command line arguments."""
    config = {
        "model": {
            "type": args.model_type,
            "latent_dim": args.latent_dim,
            "image_size": args.image_size,
            "num_channels": args.num_channels,
            "feature_map_size": 64
        },
        "training": {
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_g": args.lr_g,
            "lr_d": args.lr_d,
            "beta1": 0.5,
            "beta2": 0.999,
            "k_steps": 1,
            "save_interval": 5
        },
        "data": {
            "dataset": args.dataset,
            "data_dir": args.data_dir,
            "download": True,
            "synthetic_samples": args.synthetic_samples,
            "pattern_type": "circles"
        },
        "device": "auto",
        "logging": {
            "level": args.log_level,
            "log_file": args.log_file
        },
        "output": {
            "save_dir": args.output_dir,
            "checkpoint_dir": args.checkpoint_dir,
            "sample_dir": os.path.join(args.output_dir, "samples")
        }
    }
    return config


def get_device(config: dict) -> torch.device:
    """Get the appropriate device for training."""
    device_config = config.get("device", "auto")
    
    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS device")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU device")
    else:
        device = torch.device(device_config)
        logging.info(f"Using specified device: {device}")
    
    return device


def get_dataloader(config: dict) -> torch.utils.data.DataLoader:
    """Get the appropriate dataloader based on configuration."""
    data_config = config["data"]
    training_config = config["training"]
    
    dataset_name = data_config["dataset"]
    batch_size = training_config["batch_size"]
    image_size = config["model"]["image_size"]
    
    if dataset_name == "mnist":
        train_loader, _ = get_mnist_dataloader(
            batch_size=batch_size,
            image_size=image_size,
            download=data_config["download"],
            data_dir=data_config["data_dir"]
        )
        return train_loader
    
    elif dataset_name == "cifar10":
        train_loader, _ = get_cifar10_dataloader(
            batch_size=batch_size,
            image_size=image_size,
            download=data_config["download"],
            data_dir=data_config["data_dir"]
        )
        return train_loader
    
    elif dataset_name == "synthetic":
        return get_synthetic_dataloader(
            batch_size=batch_size,
            num_samples=data_config["synthetic_samples"],
            image_size=image_size,
            pattern_type=data_config["pattern_type"]
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logging.info(f"Loaded configuration from {args.config}")
    else:
        config = create_config_from_args(args)
        logging.info("Created configuration from command line arguments")
    
    # Setup logging
    log_config = config["logging"]
    setup_logging(
        log_level=log_config["level"],
        log_file=log_config["log_file"]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting GAN training")
    
    # Create output directories
    output_config = config["output"]
    os.makedirs(output_config["save_dir"], exist_ok=True)
    os.makedirs(output_config["checkpoint_dir"], exist_ok=True)
    os.makedirs(output_config["sample_dir"], exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_config["save_dir"], "config.json")
    save_config(config, config_path)
    
    # Get device
    device = get_device(config)
    
    # Get dataloader
    dataloader = get_dataloader(config)
    logger.info(f"Created dataloader with {len(dataloader)} batches")
    
    # Create models
    model_config = config["model"]
    generator, discriminator = get_model(
        model_type=model_config["type"],
        latent_dim=model_config["latent_dim"],
        image_size=model_config["image_size"],
        num_channels=model_config["num_channels"],
        feature_map_size=model_config["feature_map_size"]
    )
    
    logger.info(f"Created {model_config['type']} models")
    
    # Train the model
    trainer = train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        config=config["training"],
        device=device,
        save_dir=output_config["checkpoint_dir"]
    )
    
    # Generate and save final samples
    logger.info("Generating final samples")
    final_samples = trainer.generate_samples(
        num_samples=16,
        latent_dim=model_config["latent_dim"]
    )
    
    # Save samples
    sample_path = os.path.join(output_config["sample_dir"], "final_samples.png")
    trainer.plot_generated_samples(save_path=sample_path)
    
    # Save training curves
    curves_path = os.path.join(output_config["save_dir"], "training_curves.png")
    trainer.plot_training_curves(save_path=curves_path)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {output_config['save_dir']}")


if __name__ == "__main__":
    main()
