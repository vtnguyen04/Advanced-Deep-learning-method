import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader

class LEOMetaLearner(nn.Module):
    def __init__(self, 
                 input_dim: int = 1, 
                 hidden_dim: int = 64, 
                 latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

        total_params = (
            input_dim * hidden_dim +  # fc1 weight
            hidden_dim +             # fc1 bias
            hidden_dim * hidden_dim +# fc2 weight
            hidden_dim +             # fc2 bias
            hidden_dim * 1 +         # fc3 weight
            1                        # fc3 bias
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, total_params)
        )
        # Initialize decoder's last layer
        nn.init.xavier_normal_(self.decoder[-1].weight, gain=1e-2)
        nn.init.zeros_(self.decoder[-1].bias)

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def encode(self, 
        support_input: torch.Tensor):
        stats = self.encoder(support_input).mean(dim = 0)
        mean, logvar = torch.chunk(stats, 2, dim = 0)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode_params(self, z: torch.Tensor):
        params = self.decoder(z)
        split_sizes = [
            self.input_dim * self.hidden_dim,  # fc1 weight
            self.hidden_dim,                  # fc1 bias
            self.hidden_dim * self.hidden_dim,# fc2 weight
            self.hidden_dim,                  # fc2 bias
            self.hidden_dim * 1,              # fc3 weight
            1                                 # fc3 bias
        ]

        decomposed = torch.split(params, split_sizes, dim=0)
        
        return OrderedDict([
            ('layers.fc1.weight', decomposed[0].view(self.hidden_dim, self.input_dim)),
            ('layers.fc1.bias', decomposed[1]),
            ('layers.fc2.weight', decomposed[2].view(self.hidden_dim, self.hidden_dim)),
            ('layers.fc2.bias', decomposed[3]),
            ('layers.fc3.weight', decomposed[4].view(1, self.hidden_dim)),
            ('layers.fc3.bias', decomposed[5])
        ])

    def forward(self, x: torch.Tensor, params: OrderedDict):
        x = F.linear(x, params['layers.fc1.weight'], params['layers.fc1.bias'])
        x = F.relu(x)
        x = F.linear(x, params['layers.fc2.weight'], params['layers.fc2.bias'])
        x = F.relu(x)
        x = F.linear(x, params['layers.fc3.weight'], params['layers.fc3.bias'])
        return x

class LEOTrainer:
    def __init__(
        self,
        model: LEOMetaLearner,
        lr: float = 3e-4,
        inner_lr: float = 1e-4,
        inner_steps: int = 5,
        device: str = 'cpu',
        kl_weight: float = 0.1,
        gradient_clip: float = 0.25,
        inner_gradient_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.lr = lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        self.kl_weight = kl_weight
        self.gradient_clip = gradient_clip
        self.inner_gradient_clip = inner_gradient_clip

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = lr, weight_decay = 1e-5)
        self.loss_fn = nn.MSELoss()

    def adapt(
        self, 
        support_input: torch.Tensor, 
        support_output: torch.Tensor, 
        num_adapt_steps = None
    ):
        
        with torch.set_grad_enabled(True):
            mean, logvar = self.model.encode(support_input)
        z = self.model.reparameterize(mean, logvar).requires_grad_()
        num_adapt_steps = num_adapt_steps or self.inner_steps

        # Inner loop với gradient retention
        for _ in range(num_adapt_steps):
            params = self.model.decode_params(z)

            with torch.set_grad_enabled(True):
                preds = self.model(support_input, params)
                loss = self.loss_fn(preds, support_output)
            
            grad_z = torch.autograd.grad(
                loss,
                z,
                create_graph = True,  # Giữ graph cho meta-gradient
                retain_graph = True
            )[0]
            
            if self.inner_gradient_clip:
                grad_z = torch.clamp(grad_z, 
                    -self.inner_gradient_clip, 
                    self.inner_gradient_clip
                )
            
            with torch.no_grad():
                z -= self.inner_lr * grad_z  
            z.retain_grad()  
                    
        return z, mean, logvar

    def meta_train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            tasks = _split_batch(batch)
            meta_loss = 0.0
            total_kl = 0.0

            self.optimizer.zero_grad()
            
            for task in tasks:
                s_input = task['support_input'].to(self.device)
                s_target = task['support_output'].to(self.device)
                q_input = task['query_input'].to(self.device)
                q_target = task['query_target'].to(self.device)

                # Adaptation
                z, mean, logvar = self.adapt(s_input, s_target)
                params = self.model.decode_params(z)
                
                q_pred = self.model(q_input, params)
                task_loss = self.loss_fn(q_pred, q_target)
                kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                total_task_loss = task_loss + self.kl_weight * kl_div
                
                total_task_loss.backward()
                meta_loss += task_loss.item()
                total_kl += kl_div.item()

            # Clipping gradient meta-optimization
            if self.gradient_clip:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm = self.gradient_clip
                )
                        
            self.optimizer.step()
            total_loss += meta_loss / len(tasks)
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

    def infer(
        self, 
        x_plot: torch.Tensor, 
        x_support: torch.Tensor, 
        y_support: torch.Tensor, 
        x_query: torch.Tensor,
        num_adapt_steps: int = 10
    ):
        self.model.eval()
        with torch.enable_grad():
            z, _, _ = self.adapt(x_support, y_support, num_adapt_steps)
        params = self.model.decode_params(z)
        
        with torch.no_grad():
            y_pred_initial = self.model(x_plot, params)
            q_pred = self.model(x_query, params)
        return y_pred_initial, y_pred_initial, q_pred
    
def _split_batch(batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    num_tasks = batch['support_input'].size(0)
    return [{
        'support_input': batch['support_input'][i],
        'support_output': batch['support_output'][i],
        'query_input': batch['query_input'][i],
        'query_target': batch['query_target'][i]
    } for i in range(num_tasks)]