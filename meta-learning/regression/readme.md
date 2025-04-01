# Meta-Learning Framework for Sinusoidal Regression

A flexible meta-learning framework implementing MAML (Model-Agnostic Meta-Learning) and LEO (Latent Embedding Optimization) approaches for few-shot sinusoidal regression tasks.

## Overview

This project provides implementations of state-of-the-art meta-learning algorithms for sinusoidal function regression. The framework allows researchers and practitioners to experiment with different meta-learning methods to quickly adapt to new tasks with minimal training data.

## Features

- Implementation of two meta-learning algorithms:
  - MAML (Model-Agnostic Meta-Learning)
  - LEO (Latent Embedding Optimization)
- Sinusoidal task generation with configurable parameters
- Comprehensive visualization tools for model performance analysis
- Experiment tracking and result management
- TensorBoard integration
- Detailed system information logging for reproducibility
- Checkpoint saving and loading functionality

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- tqdm
- psutil (for system monitoring)

```bash

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── main.py                  # Main script to run experiments
├── config.py                # Configuration settings
├── maml.py                  # MAML implementation
├── leo.py                   # LEO implementation
├── dataset.py               # Sinusoidal task dataset
├── visualization.py         # Plotting and visualization utilities
├── results_manager.py       # Experiment tracking and management
├── requirements.txt         # Project dependencies
└── experiments/             # Saved experiment results
```

## Usage

### Running an Experiment

To run an experiment with the default configuration:

```bash
# Run with MAML
python main.py --method maml

# Run with LEO
python main.py --method leo
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--method` | Meta-learning method to use ('maml' or 'leo') | 'maml' |

### Configuration

The `Config` class in `config.py` contains all the hyperparameters and settings for the experiments. Key parameters include:

- `RANDOM_SEED`: Seed for reproducibility
- `EPOCHS`: Number of meta-training epochs
- `META_BATCH_SIZE`: Number of tasks per batch
- `SUPPORT_SAMPLES`: Number of examples in the support set
- `QUERY_SAMPLES`: Number of examples in the query set
- `INNER_STEPS`: Number of gradient updates for task adaptation
- `META_LEARNING_RATE`: Outer loop learning rate
- `INNER_LEARNING_RATE`: Inner loop learning rate
- `INPUT_DIM`: Input dimension
- `HIDDEN_DIM`: Hidden layer dimension
- `NUM_TASKS`: Total number of tasks to generate

## Implemented Methods

### MAML (Model-Agnostic Meta-Learning)

MAML learns an initialization that can be quickly adapted to new tasks with a few gradient steps. The implementation includes:

- Inner loop optimization for task adaptation
- Meta-optimization across tasks
- Gradient clipping for stability

### LEO (Latent Embedding Optimization)

LEO performs meta-learning in a low-dimensional latent space of model parameters. Features include:

- Encoder-decoder architecture for parameter generation
- Variational inference in latent space
- KL regularization for better generalization

## Experiment Tracking

The `ExperimentManager` class provides comprehensive experiment tracking:

- Automatic experiment directory creation with timestamps
- TensorBoard logging of metrics
- Saving of model checkpoints
- Results visualization storage
- System information logging for reproducibility

## Visualization Tools

The framework includes visualization utilities to help understand model performance:

- Training loss curves
- Task prediction visualization
- Performance analysis across multiple tasks
- Adaptive learning visualization

## Results Analysis

After running an experiment, you can find detailed results in the experiment directory:

```
experiments/
└── MAML_Sinusoidal_TIMESTAMP/
    ├── checkpoints/            # Saved model weights
    ├── plots/                  # Generated visualizations
    ├── tensorboard/            # TensorBoard logs
    ├── config.json             # Experiment configuration
    ├── metrics.json            # Training and evaluation metrics
    └── full_report.json        # Comprehensive experiment report
```

![task_predictions](https://github.com/user-attachments/assets/f53ef36d-860f-4983-8874-10a08adaddbc)

## Extending the Framework

### Adding a New Meta-Learning Algorithm

To add a new meta-learning algorithm:

1. Create a new file (e.g., `new_method.py`) with your model and trainer implementation
2. Update `main.py` to import and use your new method
3. Add your method to the command-line arguments parser

### Creating Custom Tasks

To implement custom task distributions beyond sinusoidal functions:

1. Extend the `SinusoidalTaskGenerator` class in `dataset.py` with your custom task logic
2. Create a corresponding dataset class that generates your custom tasks

## License

[MIT License](LICENSE)

## References

1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In ICML.
2. Rusu, A. A., Rao, D., Sygnowski, J., Vinyals, O., Pascanu, R., Osindero, S., & Hadsell, R. (2019). Meta-Learning with Latent Embedding Optimization. In ICLR.

## Acknowledgements

This project builds on research from the meta-learning community and uses PyTorch as its deep learning framework.
