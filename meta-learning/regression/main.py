import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
import platform
import subprocess

# Import custom modules
from config import Config
from maml import (
    MAMLTrainer, MamlLearner)

from leo import LEOMetaLearner, LEOTrainer

from dataset import SinusoidalTaskDataset, SinusoidalTaskGenerator
from visualization import (
    plot_training_losses,
    visualize_task_prediction, 
    analyze_model_performance
)

from tqdm.auto import tqdm
import json
from typing import Dict
from results_manager import ExperimentManager
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sinusoidal Experiment")
    parser.add_argument("--method", type = str, default = "maml", help="Method to use: 'maml' or 'leo'")
    return parser.parse_args()

def main():

    args = parse_args()

    results_manager = ExperimentManager(
        experiment_name = f"MAML_Sinusoidal",
        enable_tensorboard = True
    )
    
    try:        
        results_manager.log_params(Config.to_dict())
        torch.manual_seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        
        print("\n⚙️ Experiment Configuration:")
        Config.display()

        print("\n📦 Preparing datasets and model...")
        dataset = SinusoidalTaskDataset(
            num_tasks = Config.NUM_TASKS,
            support_samples = Config.SUPPORT_SAMPLES,
            query_samples = Config.QUERY_SAMPLES,
            device = Config.DEVICE
        )
        dataloader = DataLoader(dataset, batch_size = Config.META_BATCH_SIZE, shuffle = True)
        
        if args.method == "maml":
            best_model_path = "experiments/MAML_Sinusoidal_20250330_015404/checkpoints/best_model_epoch.pt"
            model = MamlLearner(
                input_dim = Config.INPUT_DIM,
                hidden_dim = Config.HIDDEN_DIM
            ).to(Config.DEVICE)

            trainer = MAMLTrainer(
            model,
            lr = Config.META_LEARNING_RATE,
            inner_lr = Config.INNER_LEARNING_RATE,
            meta_batch_size = Config.META_BATCH_SIZE,
            inner_steps = Config.INNER_STEPS,
            device = Config.DEVICE,
            clip_grad_norm = 1.0,
            inner_clip_norm = 0.5
        )
            
        elif args.method == "leo":
            best_model_path = "experiments/MAML_Sinusoidal_20250330_135055/checkpoints/best_model_epoch.pt"
            model = LEOMetaLearner(
                input_dim = Config.INPUT_DIM,
                hidden_dim = Config.HIDDEN_DIM, 
                latent_dim = 16
            )
            
            trainer = LEOTrainer(
                model,
                lr = Config.META_LEARNING_RATE,
                inner_lr = Config.INNER_LEARNING_RATE,
                gradient_clip = 0.5,
                kl_weight = 0.3
            )

        try:
            # # Load checkpoint
            # checkpoint = torch.load(best_model_path, map_location = Config.DEVICE)
            # # Load model weights
            # model.load_state_dict(checkpoint['model_state'])

            trainer.load_checkpoint(best_model_path)

            print(f"✅ Successfully loaded best model from {best_model_path}")

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("❗ Initializing new model from scratch")
        print("\n🎯 Starting meta-training...")
        

        # training
        def train():
            training_losses = []
            best_loss = float('inf')
            with tqdm(range(Config.EPOCHS), desc = "Meta-Training", disable = False) as epoch_pbar:
                for epoch in epoch_pbar:

                    epoch_loss = trainer.meta_train_epoch(dataloader)
                    training_losses.append(epoch_loss)
                    
                    results_manager.log_metrics(
                        {"meta_loss": epoch_loss},
                        step = epoch,
                        phase = "train"
                    )
                    
                    epoch_pbar.set_postfix({
                        'Epoch Loss': f'{epoch_loss:.4f}',
                        'Avg Loss': f'{np.mean(training_losses):.4f}'
                    })

                    if epoch % 10 == 0 or epoch == Config.EPOCHS - 1:
                        results_manager.save_model(
                            model,
                            name = "model",
                            metadata = {"epoch": epoch, "loss": epoch_loss},
                            epoch = epoch
                        )
                    
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        trainer.save_checkpoint("best.pt")
                        results_manager.save_model(
                            model,
                            name = "best_model",
                            metadata = {"epoch": epoch, "loss": epoch_loss}
                        )
            return training_losses
        training_losses = train()

        print("\n📊 Analyzing training results...")
        train_fig = plot_training_losses(training_losses)
        results_manager.save_plot(train_fig, "training_curve")
        
        print("\n🧪 Evaluating model performance...")
        task_generator = SinusoidalTaskGenerator()
        performance_results = analyze_model_performance(
            trainer = trainer,
            task_generator = task_generator,
            num_tasks = 10,
            model_type = args.method,
            num_adapt_steps = 10
        )
        
        results_manager.save_plot(performance_results['visualization'], "performance_analysis")
        results_manager.log_metrics(
            flatten_metrics(performance_results['metrics']), 
            phase = "eval"
        )
        
        print("\n🎨 Saving example predictions...")
        prediction_fig = visualize_task_prediction(
            task_generator = task_generator,
            trainer = trainer,
            num_tasks = 3,
            model_type = args.method,
            num_adapt_steps = 10,
        )
        results_manager.save_plot(prediction_fig, "task_predictions")

        print("\n📄 Generating final report...")
        generate_comprehensive_report(results_manager)
        
        print(f"\n✅ Experiment completed! Results saved to: {results_manager.experiment_dir}")

    except Exception as e:
        results_manager.logger.error(f"Experiment failed: {str(e)}")
        raise

def flatten_metrics(metrics: Dict) -> Dict:
    """Chuyển đổi metrics dạng nested thành flat"""
    flat_metrics = {}
    for category, values in metrics.items():
        for metric, stats in values.items():
            for stat, value in stats.items():
                flat_metrics[f"{category}_{metric}_{stat}"] = value
    return flat_metrics

def generate_comprehensive_report(manager: 'ExperimentManager'):
    """Tạo báo cáo tổng hợp với tất cả thông tin"""
    report_content = {
        "training_metrics": manager.metric_history.get("train", []),
        "evaluation_metrics": manager.metric_history.get("eval", []),
        "artifacts": manager.artifacts,
        "config": manager.log_params(Config.to_dict()),
        "system_info": get_system_info()  # Hàm lấy thông tin hệ thống
    }
    
    report_path = manager.experiment_dir / "full_report.json"
    with open(report_path, "w") as f:
        json.dump(report_content, f, indent=4)

def get_system_info() -> Dict[str, str]:
    """Thu thập thông tin hệ thống chi tiết cho reproducibility
    
    Returns:
        Dictionary chứa các thông tin:
        - Python version
        - PyTorch version
        - OS info
        - CPU info
        - GPU info (nếu có)
        - Memory usage
        - Command line arguments
    """
    info = {}
    
    try:
        # Python và package info
        info["python_version"] = sys.version
        info["pytorch_version"] = torch.__version__
        
        # Hệ điều hành
        info["os"] = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine()
        }
        
        # CPU info
        try:
            import psutil
            info["cpu"] = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "usage_percent": psutil.cpu_percent()
            }
        except ImportError:
            pass
        
        # GPU info
        info["gpu"] = {}
        if torch.cuda.is_available():
            info["gpu"]["device_name"] = torch.cuda.get_device_name(0)
            info["gpu"]["cuda_version"] = torch.version.cuda
            info["gpu"]["memory_allocated"] = f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
            info["gpu"]["memory_reserved"] = f"{torch.cuda.memory_reserved()/1024**3:.2f} GB"
        
        # Memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory"] = {
                "total": f"{mem.total/1024**3:.2f} GB",
                "available": f"{mem.available/1024**3:.2f} GB",
                "used_percent": f"{mem.percent}%"
            }
        except ImportError:
            pass
        
        # Command line arguments
        info["command_line"] = " ".join(sys.argv)
        
        # Installed packages (yêu cầu pip)
        try:
            from pip._internal.operations import freeze
            info["packages"] = list(freeze.freeze())
        except:
            try:
                info["packages"] = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().splitlines()
            except:
                pass

    except Exception as e:
        print(f"Error collecting system info: {str(e)}")
    
    return info

if __name__ == "__main__":
    main()