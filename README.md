# GAN Image Generation Project

A well-structured implementation of Generative Adversarial Networks (GANs) for image generation, featuring both simple GANs and Deep Convolutional GANs (DCGANs).

## Features

- **Multiple GAN Architectures**: Simple GAN for MNIST and DCGAN for higher resolution images
- **Modern Implementation**: Type hints, comprehensive documentation, and PEP8 compliance
- **Multiple Datasets**: Support for MNIST, CIFAR-10, and synthetic datasets
- **Web Interface**: Streamlit-based UI for easy interaction and visualization
- **Comprehensive Testing**: Unit tests for all core functionality
- **Configuration Management**: JSON-based configuration system
- **Logging**: Structured logging throughout the application
- **Visualization**: Training curves, sample generation, and dataset previews

## 📁 Project Structure

```
0225_Image_generation_with_GANs/
├── src/
│   ├── models/
│   │   └── dcgan.py              # GAN model implementations
│   ├── training/
│   │   └── trainer.py            # Training utilities and loops
│   └── utils/
│       └── data_utils.py         # Data handling and utilities
├── web_app/
│   └── app.py                    # Streamlit web interface
├── config/
│   └── default_config.json       # Default configuration
├── tests/
│   └── test_gan.py               # Unit tests
├── train.py                      # Main training script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/GAN-Image-Generation-Project.git
   cd GAN-Image-Generation-Project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Command Line Training

**Train a simple GAN on MNIST**:
```bash
python train.py --model-type simple --dataset mnist --epochs 30
```

**Train a DCGAN on CIFAR-10**:
```bash
python train.py --model-type dcgan --dataset cifar10 --image-size 64 --num-channels 3 --epochs 50
```

**Train on synthetic data**:
```bash
python train.py --model-type simple --dataset synthetic --epochs 20
```

### Web Interface

**Launch the Streamlit app**:
```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501` to access the interactive interface.

## Usage Examples

### Basic Training

```python
from src.models.dcgan import get_model
from src.training.trainer import train_gan
from src.utils.data_utils import get_mnist_dataloader

# Create models
generator, discriminator = get_model(
    model_type="simple",
    latent_dim=100,
    image_size=28,
    num_channels=1
)

# Get data
train_loader, _ = get_mnist_dataloader(batch_size=128)

# Train
config = {
    "num_epochs": 30,
    "batch_size": 128,
    "lr_g": 0.0002,
    "lr_d": 0.0002
}

trainer = train_gan(generator, discriminator, train_loader, config, device)
```

### Generate Images

```python
# Generate samples
samples = trainer.generate_samples(num_samples=16)

# Visualize
trainer.plot_generated_samples(save_path="samples.png")
```

### Custom Configuration

```python
import json
from src.utils.data_utils import load_config, save_config

# Load configuration
config = load_config("config/default_config.json")

# Modify parameters
config["training"]["num_epochs"] = 50
config["model"]["latent_dim"] = 128

# Save modified configuration
save_config(config, "config/custom_config.json")
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/

# Run specific test file
python -m pytest tests/test_gan.py -v
```

## Model Architectures

### Simple GAN (for MNIST)
- **Generator**: Fully connected layers (100 → 256 → 512 → 1024 → 784)
- **Discriminator**: Fully connected layers (784 → 512 → 256 → 1)
- **Activation**: ReLU/LeakyReLU + Tanh/Sigmoid

### DCGAN (for higher resolution)
- **Generator**: Transposed convolutions with batch normalization
- **Discriminator**: Convolutions with batch normalization
- **Architecture**: Follows DCGAN paper guidelines

## Configuration

The project uses JSON configuration files. Key parameters:

```json
{
  "model": {
    "type": "simple",
    "latent_dim": 100,
    "image_size": 28,
    "num_channels": 1
  },
  "training": {
    "num_epochs": 30,
    "batch_size": 128,
    "lr_g": 0.0002,
    "lr_d": 0.0002
  },
  "data": {
    "dataset": "mnist",
    "data_dir": "./data"
  }
}
```

## Web Interface Features

The Streamlit interface provides:

- **Image Generation**: Load trained models and generate new images
- **Dataset Preview**: Visualize training data
- **Model Information**: Architecture details and training tips
- **Interactive Parameters**: Adjust generation parameters in real-time
- **Download Options**: Save generated images

## Logging

The project includes comprehensive logging:

```python
import logging
from src.utils.data_utils import setup_logging

# Setup logging
setup_logging(log_level="INFO", log_file="logs/training.log")
```

## 🔧 Development

### Code Style
- Follow PEP8 guidelines
- Use type hints throughout
- Comprehensive docstrings
- Black formatting

### Running Linters
```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

## Advanced Features

### Custom Datasets
Create custom datasets by extending the `SyntheticDataset` class:

```python
class CustomDataset(SyntheticDataset):
    def _generate_custom_pattern(self):
        # Your custom pattern generation
        pass
```

### Model Evaluation
Use built-in evaluation metrics:

```python
from src.utils.data_utils import calculate_fid_score

# Calculate FID score
fid = calculate_fid_score(real_features, fake_features)
```

### Checkpoint Management
Save and load training checkpoints:

```python
# Save checkpoint
trainer.save_checkpoint(epoch, save_dir)

# Load checkpoint
trainer.load_checkpoint(checkpoint_path)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Training Instability**: Adjust learning rates or use different architectures
3. **Poor Quality Images**: Increase training epochs or improve architecture

### Performance Tips

1. **Use GPU**: Significantly faster training
2. **Batch Size**: Larger batches often improve stability
3. **Learning Rates**: Balance generator and discriminator learning rates
4. **Architecture**: DCGAN generally produces better results than simple GANs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original GAN paper by Goodfellow et al.
- DCGAN paper by Radford et al.
- PyTorch team for the excellent framework
- Streamlit team for the web interface framework

## Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation
- Review the test cases for usage examples


# GAN-Image-Generation-Project
