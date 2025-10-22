"""
Streamlit web interface for GAN image generation.

This module provides a user-friendly web interface for training GANs,
generating images, and visualizing results.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import sys
from pathlib import Path
import json
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.dcgan import get_model
from src.training.trainer import GANTrainer
from src.utils.data_utils import (
    get_mnist_dataloader,
    get_cifar10_dataloader,
    get_synthetic_dataloader,
    visualize_dataset_samples,
    setup_logging
)


def setup_page():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="GAN Image Generation",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎨 GAN Image Generation")
    st.markdown("Generate realistic images using Generative Adversarial Networks")


def get_device():
    """Get the appropriate device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(model_path: str, config: dict, device: torch.device):
    """Load a trained GAN model."""
    try:
        # Create models
        model_config = config["model"]
        generator, discriminator = get_model(
            model_type=model_config["type"],
            latent_dim=model_config["latent_dim"],
            image_size=model_config["image_size"],
            num_channels=model_config["num_channels"],
            feature_map_size=model_config["feature_map_size"]
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        
        generator.to(device)
        generator.eval()
        
        return generator, checkpoint
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def generate_images(generator, num_images: int, latent_dim: int, device: torch.device):
    """Generate images using the trained generator."""
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, device=device)
        generated_images = generator(noise)
    
    return generated_images


def tensor_to_image(tensor, denormalize=True):
    """Convert PyTorch tensor to PIL Image."""
    if denormalize:
        tensor = (tensor + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    if tensor.dim() == 4:  # Batch
        tensor = tensor.squeeze(0)
    
    if tensor.size(0) == 1:  # Grayscale
        img_array = tensor.squeeze(0).cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array, mode='L')
    else:  # RGB
        img_array = tensor.permute(1, 2, 0).cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array, mode='RGB')


def plot_loss_curves(g_losses, d_losses):
    """Plot training loss curves using Plotly."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=g_losses,
        mode='lines',
        name='Generator Loss',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        y=d_losses,
        mode='lines',
        name='Discriminator Loss',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Training Loss Curves',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified'
    )
    
    return fig


def main():
    """Main Streamlit application."""
    setup_page()
    
    # Sidebar for model selection and parameters
    st.sidebar.header("Model Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["simple", "dcgan"],
        help="Choose between simple GAN (for MNIST) or DCGAN (for higher resolution)"
    )
    
    # Dataset selection
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["mnist", "cifar10", "synthetic"],
        help="Choose the dataset to train on"
    )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    latent_dim = st.sidebar.slider("Latent Dimension", 50, 200, 100)
    image_size = st.sidebar.slider("Image Size", 28, 128, 28 if dataset == "mnist" else 64)
    num_channels = st.sidebar.selectbox("Channels", [1, 3], index=0 if dataset == "mnist" else 1)
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    num_images = st.sidebar.slider("Number of Images", 1, 25, 16)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get device
    device = get_device()
    st.sidebar.info(f"Using device: {device}")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Generate Images", "Dataset Preview", "Training", "Model Info"])
    
    with tab1:
        st.header("Image Generation")
        
        # Check if we have a trained model
        checkpoint_dir = "./checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            
            if checkpoint_files:
                selected_checkpoint = st.selectbox(
                    "Select Model Checkpoint",
                    checkpoint_files,
                    help="Choose a trained model to generate images"
                )
                
                if st.button("Load Model"):
                    checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)
                    
                    # Create config based on UI parameters
                    config = {
                        "model": {
                            "type": model_type,
                            "latent_dim": latent_dim,
                            "image_size": image_size,
                            "num_channels": num_channels,
                            "feature_map_size": 64
                        }
                    }
                    
                    generator, checkpoint = load_model(checkpoint_path, config, device)
                    
                    if generator is not None:
                        st.success("Model loaded successfully!")
                        
                        # Generate images
                        if st.button("Generate Images"):
                            with st.spinner("Generating images..."):
                                generated_images = generate_images(
                                    generator, num_images, latent_dim, device
                                )
                            
                            # Display images in a grid
                            cols = st.columns(4)
                            for i, img_tensor in enumerate(generated_images):
                                with cols[i % 4]:
                                    img_pil = tensor_to_image(img_tensor)
                                    st.image(img_pil, caption=f"Generated {i+1}")
                            
                            # Download option
                            if st.button("Download Images"):
                                for i, img_tensor in enumerate(generated_images):
                                    img_pil = tensor_to_image(img_tensor)
                                    img_bytes = io.BytesIO()
                                    img_pil.save(img_bytes, format='PNG')
                                    img_bytes.seek(0)
                                    
                                    st.download_button(
                                        label=f"Download Image {i+1}",
                                        data=img_bytes.getvalue(),
                                        file_name=f"generated_image_{i+1}.png",
                                        mime="image/png"
                                    )
            else:
                st.warning("No trained models found. Please train a model first.")
        else:
            st.warning("Checkpoint directory not found. Please train a model first.")
    
    with tab2:
        st.header("Dataset Preview")
        
        if st.button("Load Dataset Preview"):
            with st.spinner("Loading dataset..."):
                try:
                    if dataset == "mnist":
                        train_loader, _ = get_mnist_dataloader(
                            batch_size=16, image_size=image_size
                        )
                    elif dataset == "cifar10":
                        train_loader, _ = get_cifar10_dataloader(
                            batch_size=16, image_size=image_size
                        )
                    else:  # synthetic
                        train_loader = get_synthetic_dataloader(
                            batch_size=16, image_size=image_size
                        )
                    
                    # Get a batch of samples
                    data_iter = iter(train_loader)
                    images, labels = next(data_iter)
                    
                    # Display samples
                    cols = st.columns(4)
                    for i in range(min(16, len(images))):
                        with cols[i % 4]:
                            img_pil = tensor_to_image(images[i])
                            st.image(img_pil, caption=f"Sample {i+1}")
                    
                    st.success(f"Loaded {dataset.upper()} dataset preview")
                    
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
    
    with tab3:
        st.header("Training")
        
        st.info("""
        To train a new model, use the command line interface:
        
        ```bash
        python train.py --model-type simple --dataset mnist --epochs 30
        ```
        
        Or for DCGAN with CIFAR-10:
        
        ```bash
        python train.py --model-type dcgan --dataset cifar10 --image-size 64 --num-channels 3 --epochs 50
        ```
        """)
        
        # Training parameters
        st.subheader("Training Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs", value=30, min_value=1, max_value=200)
            batch_size = st.number_input("Batch Size", value=128, min_value=1, max_value=512)
        
        with col2:
            lr_g = st.number_input("Generator Learning Rate", value=0.0002, format="%.4f")
            lr_d = st.number_input("Discriminator Learning Rate", value=0.0002, format="%.4f")
        
        if st.button("Start Training"):
            st.warning("Training is not implemented in the web interface. Please use the command line.")
    
    with tab4:
        st.header("Model Information")
        
        st.subheader("Model Architecture")
        
        if model_type == "simple":
            st.code("""
            Generator:
            - Linear(100 -> 256) + ReLU
            - Linear(256 -> 512) + ReLU  
            - Linear(512 -> 1024) + ReLU
            - Linear(1024 -> 784) + Tanh
            
            Discriminator:
            - Linear(784 -> 512) + LeakyReLU(0.2)
            - Linear(512 -> 256) + LeakyReLU(0.2)
            - Linear(256 -> 1) + Sigmoid
            """)
        else:
            st.code("""
            Generator (DCGAN):
            - ConvTranspose2d layers with BatchNorm and ReLU
            - Upsampling from 4x4 to 64x64
            - Final Tanh activation
            
            Discriminator (DCGAN):
            - Conv2d layers with BatchNorm and LeakyReLU
            - Downsampling from 64x64 to 4x4
            - Final Sigmoid activation
            """)
        
        st.subheader("Training Details")
        st.markdown("""
        - **Loss Function**: Binary Cross Entropy (BCE)
        - **Optimizer**: Adam with β1=0.5, β2=0.999
        - **Learning Rate**: 0.0002 (both networks)
        - **Batch Size**: 128
        - **Epochs**: 30-50 depending on dataset
        """)
        
        st.subheader("Tips for Better Results")
        st.markdown("""
        1. **Stable Training**: Use proper weight initialization
        2. **Learning Rates**: Keep discriminator and generator learning rates balanced
        3. **Batch Size**: Larger batch sizes often lead to more stable training
        4. **Epochs**: Train for enough epochs to see convergence
        5. **Architecture**: DCGAN generally produces better results than simple GANs
        """)


if __name__ == "__main__":
    main()
