# CIFAR-10 Image Classifier with PyTorch Lightning

A flexible and easy-to-use image classification framework built with PyTorch Lightning for training various deep learning architectures on the CIFAR-10 dataset.
## Features

- Support for multiple model architectures (AlexNet, ResNet, VGG, etc.)
- Training with pre-trained models
- Custom visualization of model predictions
- GPU acceleration support
- TensorBoard logging
- Early stopping and model checkpointing
- Beautiful progress tracking with colorful console output

<!-- Hiển thị 2 hình ảnh trên cùng một hàng -->
<div align="center">
    <img src="https://github.com/user-attachments/assets/6ccbe959-a4e9-4a20-9f4d-95b2526bdde0" width="65%" />
    <img src="https://github.com/user-attachments/assets/f16b24d2-1eef-4a4a-be16-ae5616c30134" width="30%" />
</div>

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.7+
- PyTorch Lightning
- CUDA (optional, for GPU acceleration)
```bash
# Install dependencies
pip install -r requirements.txt
```

## Dependencies

The project requires the following Python packages:
- torch
- torchvision
- pytorch_lightning
- matplotlib
- tqdm
- colorama
- pyfiglet
- torchinfo
- torchmetrics
- timm

You can install all dependencies by running:

```bash
pip install requirements.txt
```

## Usage

### Training a Model

To train a model, run the `CIFAR-10_train.py` script with your desired parameters:

```bash
python CIFAR-10_train.py --model AlexNet --pretrained --batch-size 15 --epochs 5 --lr 0.001
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model architecture to use | AlexNet |
| `--pretrained` | Whether to use pre-trained weights | False |
| `--batch-size` | Batch size for training | 12 |
| `--epochs` | Number of epochs to train | 10 |
| `--lr` | Learning rate | 0.001 |
| `--no-cuda` | Disable CUDA (GPU) training | False |
| `--seed` | Random seed for reproducibility | 1 |

### Available Models

The model architectures available for training are listed in the `model_names.txt` file. Add or remove model names from this file to customize the available architectures. These architectures are continuously updated and new ones are added regularly to ensure compatibility with state-of-the-art models.

Some of the supported models include:
- AlexNet
- ResNet (various versions)
- VGG16
- DenseNet
- EfficientNet
- MobileNet
- ...

## Project Structure

```
.
├── CIFAR-10_train.py                # Main training script
├── get_models.py           # Module for retrieving model architectures
├── model_names.txt         # List of available model architectures
├── requirements.txt        # Project dependencies
├── data/                   # Directory for storing datasets
└── models/                 # Directory for storing models architechture

```

## How It Works

1. **Model Selection**: Choose from various CNN architectures with optional pre-trained weights.
2. **Data Loading**: CIFAR-10 dataset is automatically downloaded and preprocessed.
3. **Training**: The model is trained using PyTorch Lightning with custom progress tracking.
4. **Validation**: Model is validated after each epoch to monitor performance.
5. **Testing**: Final model is tested and predictions are visualized.
6. **Visualization**: Training results are logged to TensorBoard and model predictions are displayed.

## Customization

### Adding New Models

To add support for a new model architecture:

1. Add the model name to `model_names.txt`
2. Implement the model loading logic in `get_models.py`

### Modifying Data Preprocessing

You can customize the data preprocessing pipeline by modifying the `transform` in the `main()` function of `CIFAR-10_train.py`.

## Visualization

- **TensorBoard**: Training metrics are logged to TensorBoard. Run `tensorboard --logdir = runs` to view the training progress.
- **Prediction Visualization**: After training, the script visualizes model predictions on test samples.

## License

[MIT License](LICENSE)

## Acknowledgements

This project utilizes the CIFAR-10 dataset and various SOTA CNN model architectures.
