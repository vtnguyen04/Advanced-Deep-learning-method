import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm

from collections import OrderedDict
import copy
import math

class MamlLearner(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleDict({
            'fc1': nn.Linear(input_dim, hidden_dim),
            'fc2': nn.Linear(hidden_dim, hidden_dim),
            'fc3': nn.Linear(hidden_dim, 1)
        })

        for name, layer in self.layers.items():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, params: OrderedDict = None) -> torch.Tensor:
        if params is None:
            params = OrderedDict(self.named_parameters())
        
        # Layer 1
        x = F.linear(x, params['layers.fc1.weight'], params['layers.fc1.bias'])
        x = F.relu(x)
        
        # Layer 2
        x = F.linear(x, params['layers.fc2.weight'], params['layers.fc2.bias'])
        x = F.relu(x)
        # Output layer
        x = F.linear(x, params['layers.fc3.weight'], params['layers.fc3.bias'])
        return x
    
def _split_batch(batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Split batch into individual tasks"""
        tasks = []
        num_tasks = batch['support_input'].size(0)  
        
        for i in range(num_tasks):
            task = {
                'support_input': batch['support_input'][i],
                'support_output': batch['support_output'][i],
                'query_input': batch['query_input'][i],
                'query_target': batch['query_target'][i]
            }
            tasks.append(task)
        return tasks

class MAMLTrainer:
    """Model-Agnostic Meta-Learning implementation with enhanced functionality."""
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        lr: float = 0.001, 
        inner_lr: float = 0.01, 
        meta_batch_size: int = 32, 
        inner_steps: int = 5,
        device: Union[str, torch.device] = 'cpu',
        loss_fn: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,

        # New gradient clipping parameters
        clip_grad_norm: Optional[float] = 1.0,
        clip_grad_value: Optional[float] = None,
        inner_clip_norm: Optional[float] = None,
        appy_gradient_clipping: Optional[bool] = True
    ):
        self.model = model.to(device)
        self.lr = lr
        self.inner_lr = inner_lr
        self.meta_batch_size = meta_batch_size
        self.inner_steps = inner_steps
        self.device = device
        
        # Initialize optimizer and loss function
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr = lr)
        self.loss_fn = loss_fn or torch.nn.MSELoss()
        
        # Gradient clipping configurations
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.inner_clip_norm = inner_clip_norm
        self.appy_gradient_clipping = appy_gradient_clipping
        self.meta_train_losses = []

    def _apply_gradient_clipping(self):
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm = self.clip_grad_norm
            )
        if self.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                clip_value = self.clip_grad_value
            )

    def adapt(
        self, 
        support_input: torch.Tensor, 
        support_target: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """Produce task-adapted model copy."""
        num_steps = num_steps or self.inner_steps
        original_params = OrderedDict(self.model.named_parameters())
        
        for _ in range(num_steps):
            # Compute gradients
            preds = self.model(support_input, params = original_params)
            loss = self.loss_fn(preds, support_target)
            grads = torch.autograd.grad(loss, original_params.values(), create_graph = True, allow_unused = True)
            
            original_params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(original_params.items(), grads)
            )
        return original_params

    def _outer_loop(self, tasks: List[Dict[str, torch.Tensor]], training: bool = True) -> torch.Tensor:
        """Compute meta-loss across task batch"""
        task_losses = []
        
        for task in tasks:  
            s_input = task['support_input'].to(self.device)
            s_target = task['support_output'].to(self.device)
            q_input = task['query_input'].to(self.device)
            q_target = task['query_target'].to(self.device)
            
            # Inner loop adaptation
            adapted_params = self.adapt(s_input, s_target)
            
            # Compute query loss
            with torch.set_grad_enabled(training):
                q_pred = self.model(q_input, params = adapted_params)
                task_loss = self.loss_fn(q_pred, q_target)
                task_losses.append(task_loss)
        
        return torch.mean(torch.stack(task_losses))

    def meta_train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch in dataloader:

            tasks = _split_batch(batch)
            meta_loss = self._outer_loop(tasks)
            total_loss += meta_loss.item()
            
            self.optimizer.zero_grad()
            meta_loss.backward()
            if self.appy_gradient_clipping:
               self._apply_gradient_clipping()
            self.optimizer.step()
            
        return total_loss / num_batches
    
    def infer(
        self, 
        x_plot: torch.Tensor,
        x_support: torch.Tensor,
        y_support: torch.Tensor,
        x_query: torch.Tensor,
        num_steps: Optional[int] = 10
    ) -> Dict[str, float]:
        """Comprehensive inference with metrics."""
        self.model.eval()

        with torch.enable_grad():
            adapted_params = self.adapt(x_support, y_support, num_steps = num_steps)
        
        # Compute query loss
        with torch.set_grad_enabled(False):
            y_pred_initial = self.model(x_plot)
            q_pred = self.model(x_query, params = adapted_params)
            y_pred_adapted = self.model(x_plot, params = adapted_params)

        return y_pred_initial, y_pred_adapted, q_pred 

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
