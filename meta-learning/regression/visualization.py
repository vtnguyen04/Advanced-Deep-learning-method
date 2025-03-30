import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Union, Literal
from tqdm.auto import tqdm
import torch.nn.functional as F

ModelType = Literal['maml', 'leo', 'reptile']  # Có thể mở rộng thêm

def plot_training_losses(
    losses: List[float],
    model_type: ModelType = 'maml'
) -> plt.Figure:
    """Visualize training progress for any meta-learning model"""
    fig = plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    title_map = {
        'maml': 'MAML',
        'leo': 'LEO',
        'reptile': 'Reptile'
    }
    
    sns.lineplot(x = range(len(losses)), y = losses, label = 'Meta Loss')
    plt.title(f'{title_map[model_type]} Training Progress', fontsize=16)
    plt.xlabel('Epochs', fontsize = 12)
    plt.ylabel('Meta Loss', fontsize = 12)
    plt.legend()
    plt.tight_layout()
    return fig


def visualize_task_prediction(
    trainer: Union['MAML', 'LEO', 'Reptile'],
    task_generator: 'SinusoidalTaskGenerator',
    model_type: ModelType = 'maml',
    num_tasks: int = 3,
    num_adapt_steps: int = 10,
    num_plot_points: int = 300
) -> plt.Figure:
    """Unified visualization for different meta-learning models"""
    
    title_map = {
        'maml': 'MAML',
        'leo': 'LEO',
        'reptile': 'Reptile'
    }
    
    fig, axs = plt.subplots(num_tasks, 4, figsize=(25, 6*num_tasks))
    if num_tasks == 1:
        axs = [axs]
    
    for task_idx in range(num_tasks):
        task = task_generator.sample_task()
        x_plot = torch.linspace(-5, 5, num_plot_points).unsqueeze(1).to(trainer.device)
        x_support, y_support = task.sample(20, device = trainer.device)
        x_query, y_query = task.sample(10, device = trainer.device)

        # Model-specific inference
        with torch.no_grad():
            y_pred_initial, y_pred_adapted, query_pred = trainer.infer(x_plot, x_support, y_support, x_query, num_adapt_steps)
            
            y_true = task.amplitude * torch.sin(x_plot + task.phase)

        def to_numpy(tensor):
            return tensor.cpu().numpy().squeeze()
            
        plot_data = {
            'x': to_numpy(x_plot),
            'y_true': to_numpy(y_true),
            'y_initial': to_numpy(y_pred_initial),
            'y_adapted': to_numpy(y_pred_adapted),
            'x_support': to_numpy(x_support),
            'y_support': to_numpy(y_support),
            'x_query': to_numpy(x_query),
            'y_query': to_numpy(y_query),
            'query_pred': to_numpy(query_pred)
        }
        
        # Draw graphs
        axes = axs[task_idx] if num_tasks > 1 else axs
        
        # 1. Initial prediction
        axes[0].plot(plot_data['x'], plot_data['y_initial'], label = 'Prediction', linewidth=2)
        axes[0].plot(plot_data['x'], plot_data['y_true'], '--', label = 'Ground Truth', color='black')
        axes[0].set_title(f'Task {task_idx+1}: Before Adaptation')
        axes[0].legend()
        
        # 2. Prediction after adaptation
        axes[1].plot(plot_data['x'], plot_data['y_adapted'], label = 'Adapted', linewidth = 2)
        axes[1].plot(plot_data['x'], plot_data['y_true'], '--', color = 'black')
        axes[1].scatter(plot_data['x_support'], plot_data['y_support'], 
                        label = 'Support set', s = 80, edgecolor = 'white', zorder = 3)
        axes[1].scatter(plot_data['x_query'], plot_data['y_query'], 
                        marker = 'X', label = 'Query set', s = 100, edgecolor = 'white', zorder = 3)
        axes[1].set_title(f'After {num_adapt_steps} Adaptation Steps')
        axes[1].legend()
        
        # 3. Error distribution
        error = np.abs(plot_data['y_adapted'] - plot_data['y_true'])
        axes[2].plot(plot_data['x'], error, label = 'Absolute Error', color = 'purple')
        axes[2].set_yscale('log')
        axes[2].set_title('Error Distribution (log scale)')
        axes[2].set_ylabel('Error')
        axes[2].legend()
        
        # 4. Performance on query set
        axes[3].scatter(plot_data['y_query'], plot_data['query_pred'], 
                       alpha = 0.7, edgecolor = 'k', s=80)
        axes[3].plot([-5, 5], [-5, 5], '--', color = 'grey')
        axes[3].set_title('Prediction vs Ground Truth (Query set)')
        axes[3].set_xlabel('Ground Truth Value')
        axes[3].set_ylabel('Prediction')
        
        # Add task information
        task_info = (
            f"Amplitude: {task.amplitude:.2f}\n"
            f"Phase: {task.phase:.2f}\n"
            f"Seed: {task.seed}"
        )
        axes[0].text(0.05, 0.95, task_info, 
                    transform = axes[0].transAxes,
                    verticalalignment = 'top',
                    bbox=dict(facecolor = 'white', alpha = 0.9))
        
        # General formatting
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-6, 6)

    plt.tight_layout()
    return fig

def analyze_model_performance(
    trainer: Union['MAML', 'LEO', 'Reptile'],
    task_generator: 'SinusoidalTaskGenerator',
    model_type: ModelType = 'maml',
    num_tasks: int = 100,
    num_adapt_steps: int = 5
) -> Dict[str, Union[float, Dict]]:
    """Unified performance analysis for meta-learning models"""
    
    metrics = {
        'pre_adapt': {'mse': [], 'mae': []},
        'post_adapt_support': {'mse': [], 'mae': []},
        'post_adapt_query': {'mse': [], 'mae': []}
    }

    error_analysis = []

    for _ in tqdm(range(num_tasks), desc="Evaluating"):
        task = task_generator.sample_task()
        x_support, y_support = task.sample(20, trainer.device)
        x_query, y_query = task.sample(10, trainer.device)

        # Model-specific pre-adaptation
        with torch.no_grad():
            if model_type == 'leo':
                z = torch.randn(trainer.model.latent_dim).to(trainer.device)
                params = trainer.model.decode_params(z)
                y_pred = trainer.model(x_query, params)
            else:
                y_pred = trainer.model(x_query)
            
            # Thêm cả MSE và MAE cho pre-adaptation
            pre_mse = F.mse_loss(y_pred, y_query)
            pre_mae = F.l1_loss(y_pred, y_query)
            metrics['pre_adapt']['mse'].append(pre_mse.item())
            metrics['pre_adapt']['mae'].append(pre_mae.item())  # Đã thêm dòng này

        # Model-specific adaptation
        if model_type == 'maml':
            adapted_params = trainer.adapt(x_support, y_support)
        elif model_type == 'leo':
            z, _, _ = trainer.adapt(x_support, y_support)
            adapted_params = trainer.model.decode_params(z)
        elif model_type == 'reptile':
            adapted_params = trainer.adapt(x_support, y_support)

        # Unified evaluation
        with torch.no_grad():
            y_support_pred = trainer.model(x_support, adapted_params)
            y_query_pred = trainer.model(x_query, adapted_params)

            post_support_mse = F.mse_loss(y_support_pred, y_support)
            post_support_mae = F.l1_loss(y_support_pred, y_support)
            
            
            # On query set
            y_query_pred = trainer.model(x_query, adapted_params)
            post_query_mse = F.mse_loss(y_query_pred, y_query)
            post_query_mae = F.l1_loss(y_query_pred, y_query)
            
            metrics['post_adapt_support']['mse'].append(post_support_mse.item())
            metrics['post_adapt_support']['mae'].append(post_support_mae.item())
            metrics['post_adapt_query']['mse'].append(post_query_mse.item())
            metrics['post_adapt_query']['mae'].append(post_query_mae.item())
            
            # Error analysis
            error_analysis.append({
                'task_params': {'amplitude': task.amplitude, 'phase': task.phase},
                'support_error': (x_support.cpu().numpy(), (y_support_pred - y_support).cpu().numpy()),
                'query_error': (x_query.cpu().numpy(), (y_query_pred - y_query).cpu().numpy())
            })

    # Calculate aggregate metrics
    def calculate_stats(data):
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)
        }
    
    final_metrics = {
        'pre_adapt': {k: calculate_stats(v) for k, v in metrics['pre_adapt'].items()},
        'post_adapt_support': {k: calculate_stats(v) for k, v in metrics['post_adapt_support'].items()},
        'post_adapt_query': {k: calculate_stats(v) for k, v in metrics['post_adapt_query'].items()}
    }

    # Visualization
    fig = plt.figure(figsize=(18, 10))
    grid = plt.GridSpec(3, 3, figure=fig)
    
    # Error distribution
    ax1 = fig.add_subplot(grid[:2, :2])
    ax1.boxplot([
        metrics['pre_adapt']['mae'],
        metrics['post_adapt_query']['mae']
    ], labels = ['Before Adaptation', 'After Adaptation'])
    ax1.set_title('Absolute Error Distribution')
    ax1.set_ylabel('MAE')
    
    # Error analysis by task parameters
    ax2 = fig.add_subplot(grid[0, 2])
    amplitudes = [e['task_params']['amplitude'] for e in error_analysis]
    query_mae = [np.mean(np.abs(e['query_error'][1])) for e in error_analysis]
    ax2.scatter(amplitudes, query_mae, alpha=0.6)
    ax2.set_xlabel('Amplitude')
    ax2.set_ylabel('Average Error')
    ax2.set_title('Impact of Amplitude on Error')
    
    ax3 = fig.add_subplot(grid[1, 2])
    phases = [e['task_params']['phase'] for e in error_analysis]
    ax3.scatter(phases, query_mae, alpha=0.6)
    ax3.set_xlabel('Phase')
    ax3.set_title('Impact of Phase on Error')
    
    # Adaptation example
    ax4 = fig.add_subplot(grid[2, :])
    sample_task = error_analysis[-1]
    x_plot = np.linspace(-5, 5, 100)
    ax4.plot(x_plot, sample_task['task_params']['amplitude'] * np.sin(x_plot + sample_task['task_params']['phase']))
    ax4.scatter(*sample_task['support_error'], label='Support Error')
    ax4.scatter(*sample_task['query_error'], label='Query Error')
    ax4.set_title('Typical Error Distribution')
    ax4.legend()
    
    plt.tight_layout()
    
    return {
        'metrics': final_metrics,
        'visualization': fig,
        'error_analysis': error_analysis
    }