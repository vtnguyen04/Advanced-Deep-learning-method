import torch

class Config:
    """Configuration class for MAML Sinusoidal Regression"""
    # Dataset Parameters
    NUM_TASKS = 100
    SUPPORT_SAMPLES = 30
    QUERY_SAMPLES = 45

    # Model Architecture
    INPUT_DIM = 1
    HIDDEN_DIM = 40
    # Training Hyperparameters
    META_LEARNING_RATE = 1e-3
    INNER_LEARNING_RATE = 7e-4
    META_BATCH_SIZE = 100
    INNER_STEPS = 5
    EPOCHS = 500

    # Reproducibility
    RANDOM_SEED = 42

    # Device Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def display(cls):
        """Display configuration details"""
        print("Meta-Learning Project Configuration:")
        for key, value in vars(cls).items():
            if not key.startswith('__') and not callable(value):
                print(f"{key}: {value}")


    @classmethod
    def to_dict(cls):
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('__') and not callable(getattr(cls, key))
        }