import os
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

class ExperimentManager:
    """
    Advanced experiment management system for machine learning research
    
    Features:
    - Automated experiment tracking and versioning
    - Comprehensive artifact management
    - Rich visualization and reporting
    - TensorBoard integration
    - Error handling and logging
    
    Example usage:
    >>> manager = ExperimentManager("meta_learning_v1")
    >>> manager.log_params(config.__dict__)
    >>> manager.log_metrics(train_metrics)
    >>> manager.save_model(model)
    """
    
    def __init__(
        self, 
        experiment_name: Optional[str] = None,
        base_dir: str = "experiments",
        log_level: str = "INFO",
        enable_tensorboard: bool = True
    ):
        """
        Initialize experiment manager
        
        Args:
            experiment_name: Unique identifier for the experiment
            base_dir: Root directory for all experiments
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_tensorboard: Enable TensorBoard logging
        """
        
        self._setup_logging(log_level)
        self.base_dir = Path(base_dir)
        self.experiment_name = self._generate_experiment_name(experiment_name)
        self.experiment_dir = self._create_experiment_dir()
        self.artifacts = defaultdict(list)
        self.writer = None

        if enable_tensorboard:
            self.writer = SummaryWriter(log_dir=self.experiment_dir / "tensorboard")
            
        self.logger.info(f"Initialized experiment: {self.experiment_name}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")

    def _setup_logging(self, log_level: str):
        """Configure logging system"""
        self.logger = logging.getLogger("ExperimentManager")
        self.logger.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def _generate_experiment_name(self, name: Optional[str]) -> str:
        """Generate unique experiment name with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{timestamp}" if name else f"experiment_{timestamp}"

    def _create_experiment_dir(self) -> Path:
        """Create directory structure for experiment"""
        dir_path = self.base_dir / self.experiment_name
        try:
            dir_path.mkdir(parents=True, exist_ok=False)
            for subdir in ["checkpoints", "metrics", "plots", "configs"]:
                (dir_path / subdir).mkdir()
            return dir_path
        except FileExistsError:
            self.logger.error(f"Experiment directory already exists: {dir_path}")
            raise

    def log_params(self, params: Dict[str, Any]):
        """
        Log experiment parameters/hyperparameters
        
        Args:
            params: Dictionary of parameters to log
        """
        params_path = self.experiment_dir / "configs" / "params.json"
        try:
            # Chuyển đổi các giá trị không thể serialize
            serializable_params = {}
            for key, value in params.items():
                try:
                    json.dumps(value)  # Test khả năng serialize
                    serializable_params[key] = value
                except TypeError:
                    serializable_params[key] = str(value)
            
            with open(params_path, "w") as f:
                json.dump(serializable_params, f, indent=4)
            self.logger.info(f"Logged parameters to {params_path}")
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {str(e)}")
            raise

    def log_metrics(
        self, 
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        phase: str = "train"
    ):
        """
        Log training/validation metrics
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step/epoch number
            phase: Phase of experiment (train/val/test)
        """
        # Save to JSON
        metrics_path = self.experiment_dir / "metrics" / f"{phase}_metrics.json"
        try:
            existing_metrics = []
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    existing_metrics = json.load(f)
            
            existing_metrics.append({
                "step": step or len(existing_metrics),
                "metrics": metrics
            })
            
            with open(metrics_path, "w") as f:
                json.dump(existing_metrics, f, indent=4)
                
            # TensorBoard logging
            if self.writer and step is not None:
                for name, value in metrics.items():
                    self.writer.add_scalar(f"{phase}/{name}", value, step)
                    
            self.logger.debug(f"Logged {len(metrics)} {phase} metrics at step {step}")
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {str(e)}")
            raise

    def save_model(
        self,
        model: torch.nn.Module,
        name: str = "model",
        metadata: Optional[Dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None
    ):
        """
        Save model checkpoint with metadata
        
        Args:
            model: Model to save
            name: Base name for checkpoint files
            metadata: Additional metadata to include
            optimizer: Optimizer state to save
            epoch: Current training epoch
        """
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "metadata": metadata or {}
        }
        
        checkpoint_path = self.experiment_dir / "checkpoints" / f"{name}_epoch{epoch or ''}.pt"
        try:
            torch.save(checkpoint, checkpoint_path)
            self.artifacts["models"].append(str(checkpoint_path))
            self.logger.info(f"Saved model checkpoint to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise

    def save_plot(
        self, 
        fig: plt.Figure,
        name: str,
        dpi: int = 300,
        formats: List[str] = ["png", "pdf"]
    ):
        """
        Save matplotlib figure to experiment directory
        
        Args:
            fig: Matplotlib figure object
            name: Name for the plot file (without extension)
            dpi: Image resolution
            formats: List of formats to save (png, pdf, svg, etc.)
        """
        plot_dir = self.experiment_dir / "plots"
        try:
            for fmt in formats:
                save_path = plot_dir / f"{name}.{fmt}"
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                self.artifacts["plots"].append(str(save_path))
            plt.close(fig)
            self.logger.info(f"Saved plot '{name}' in {len(formats)} formats")
        except Exception as e:
            self.logger.error(f"Failed to save plot: {str(e)}")
            raise

    def generate_report(self):
        """Generate comprehensive HTML report"""
        # Implementation for generating interactive HTML report
        # Combines metrics, plots, and analysis into a single document
        pass  # Detailed implementation would be here

    def archive(self, format: str = "zip"):
        """Archive experiment directory"""
        # Implementation for archiving experiment
        pass

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context"""
        if self.writer:
            self.writer.close()
        if exc_type:
            self.logger.error(f"Experiment terminated with error: {exc_val}")
        else:
            self.logger.info("Experiment completed successfully")

    @property
    def checkpoint_dir(self) -> Path:
        """Get path to checkpoints directory"""
        return self.experiment_dir / "checkpoints"

    @property
    def metric_history(self) -> Dict:
        """Load and return metric history"""
        metrics = {}
        for phase in ["train", "val", "test"]:
            metric_file = self.experiment_dir / "metrics" / f"{phase}_metrics.json"
            if metric_file.exists():
                with open(metric_file, "r") as f:
                    metrics[phase] = json.load(f)
        return metrics

    def log_hyperparameter_search(self, results: List[Dict]):
        """Log hyperparameter search results"""
        search_path = self.experiment_dir / "configs" / "hyperparameters.json"
        with open(search_path, "w") as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Logged hyperparameter search results to {search_path}")

    def log_git_info(self):
        """Attempt to log Git repository information"""
        try:
            from git import Repo, InvalidGitRepositoryError
            repo = Repo(search_parent_directories=True)
            commit = repo.head.commit.hexsha
            diff = repo.git.diff()
            
            git_info = {
                "commit": commit,
                "diff": diff,
                "branch": repo.active_branch.name
            }
            
            with open(self.experiment_dir / "configs" / "git_info.json", "w") as f:
                json.dump(git_info, f, indent=4)
                
            self.logger.info("Logged Git repository information")
        except ImportError:
            self.logger.warning("GitPython not installed, skipping Git logging")
        except InvalidGitRepositoryError:
            self.logger.warning("Not in a Git repository, skipping Git logging")