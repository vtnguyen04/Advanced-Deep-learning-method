import torch
import numpy as np
from torch.utils.data import Dataset

from typing import Tuple, Dict
import random


class Task:
    """Represents a single sinusoidal regression task."""
    def __init__(self, amplitude: float, phase: float, seed: int = None):
        """
        Initialize a task with specific amplitude and phase.
        
        Args:
            amplitude: Amplitude of the sinusoidal function
            phase: Phase shift of the sinusoidal function
            seed: Optional seed for reproducibility
        """
        self.amplitude = amplitude
        self.phase = phase
        self.seed = seed if seed is not None else random.randint(0, 10000)
    
    @torch.no_grad()
    def sample(self, num_samples: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample data points for the task with consistent randomness.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to place the tensors
        
        Returns:
            Tuple of x and y tensors
        """
        # Set seed for reproducibility within this task
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        x = torch.linspace(-5, 5, num_samples, device=device).unsqueeze(1)
        y = self.amplitude * torch.sin(x + self.phase)
        
        # Add small gaussian noise
        # noise = torch.normal(0, 0.1, size=y.shape, device=device)
        # y += noise
        
        return x, y

class SinusoidalTaskGenerator:
    """Generator for sinusoidal regression tasks."""
    def __init__(
        self,
        amplitude_range: Tuple[float, float] = (0.1, 5.0),
        phase_range: Tuple[float, float] = (0, np.pi),
        seed: int = None
    ):
        """
        Initialize task generator with configurable ranges.
        
        Args:
            amplitude_range: Range for task amplitudes
            phase_range: Range for task phase shifts
            seed: Optional seed for reproducibility
        """
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.seed = seed if seed is not None else random.randint(0, 10000)
        
        # Set seed for numpy and random
        np.random.seed(self.seed)
        random.seed(self.seed)
    
    def sample_task(self, seed: int = None) -> Task:
        """
        Generate a random sinusoidal task.
        
        Args:
            seed: Optional seed for task generation
        
        Returns:
            Task instance
        """
        # Use provided seed or generate a new one
        task_seed = seed if seed is not None else random.randint(0, 10000)
        
        # Set seed for reproducibility within this method
        np.random.seed(task_seed)
        
        amplitude = np.random.uniform(*self.amplitude_range)
        phase = np.random.uniform(*self.phase_range)
        
        return Task(amplitude, phase, seed = task_seed)

class SinusoidalTaskDataset(Dataset):
    """Enhanced dataset for generating meta-learning tasks with reproducible indexing."""
    def __init__(
        self, 
        num_tasks: int = 1000, 
        support_samples: int = 20, 
        query_samples: int = 10,
        device: str = 'cpu',
        seed: int = None
    ):
        """
        Initialize the dataset with configurable parameters.
        
        Args:
            num_tasks: Total number of tasks to generate
            support_samples: Number of support samples per task
            query_samples: Number of query samples per task
            device: Device to place tensors
            seed: Optional seed for reproducibility
        """
        self.num_tasks = num_tasks
        self.task_generator = SinusoidalTaskGenerator(seed=seed)
        self.support_samples = support_samples
        self.query_samples = query_samples
        self.device = device
        self.seed = seed if seed is not None else random.randint(0, 10000)
        
        # Pre-generate task seeds for consistent indexing
        np.random.seed(self.seed)
        self.task_seeds = np.random.randint(0, 10000, size = num_tasks)
    
    def __len__(self) -> int:
        """Return total number of tasks."""
        return self.num_tasks
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a specific task by index with consistent task generation.
        
        Args:
            idx: Index of the task
        
        Returns:
            Dictionary containing support and query data for the task
        """
        # Ensure index is within range
        idx = idx % self.num_tasks
        
        # Use the pre-generated seed for this specific index
        task_seed = self.task_seeds[idx]
        
        # Generate task with the specific seed
        task = self.task_generator.sample_task(seed = task_seed)
        
        # Sample support and query data
        support_input, support_output = task.sample(self.support_samples, self.device)
        query_input, query_target = task.sample(self.query_samples, self.device)
        
        return {
            'support_input': support_input,
            'support_output': support_output,
            'query_input': query_input,
            'query_target': query_target,
            'amplitude': task.amplitude,
            'phase': task.phase
        }