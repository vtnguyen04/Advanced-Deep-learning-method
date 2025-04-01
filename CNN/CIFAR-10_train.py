#%%

""" Import Dependencies """
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import Accuracy
import matplotlib.pyplot as plt
from torchinfo import summary
from get_models import get_model
from tqdm import tqdm
from colorama import Fore, Back, Style, init
import pyfiglet

# Khởi tạo colorama
init(autoreset = True)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
""" Build Model """
class ImageClassifierModel(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            num_classes: int,
            lr: float,
            pretrained: bool = False
        ):

        super().__init__()
        self.model = get_model(
            model_name, 
            num_classes, 
            pretrained = pretrained
        )
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = Accuracy(
            task = "multiclass", 
            num_classes = num_classes
        )

    def forward(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        
        return self.model(inputs)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        
        X, y = batch
        outputs = self(X)
        loss = self.criterion(outputs, y)
        acc = self.accuracy(outputs, y)

        self.log(
            'train_loss', 
            loss, 
            on_step = True, 
            on_epoch = True, 
            prog_bar = True
        )
        self.log(
            'train_acc', 
            acc, 
            on_epoch = True, 
            prog_bar = True
        )
        return {
            'loss': loss, 
            'train_acc': acc
        }
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        
        X, y = batch
        outputs = self(X)
        loss = self.criterion(outputs, y)
        acc = self.accuracy(outputs, y)
        self.log(
            'val_loss', 
            loss, 
            on_epoch = True, 
            prog_bar = True
        )
        self.log(
            'val_acc', 
            acc, 
            on_epoch = True, 
            prog_bar = True
        )
        return {
            'val_loss': loss, 
            'val_acc': acc
        }
    
    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        acc = self.accuracy(outputs, y)
        
        self.log(
            'test_loss', 
            loss, 
            on_epoch = True
        )
        self.log(
            'test_acc', 
            acc, 
            on_epoch = True
        )
        return {
            'test_loss': loss, 
            'test_acc': acc
        }

    def configure_optimizers(self) -> Dict[str, object]:

        optimizer = optim.AdamW(self.parameters(), 
                                lr = self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = 'min', 
            factor = 0.1, 
            patience = 5, 
            verbose = True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
        
#%%
""" Show predicted results """

def show_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    num_images: int = 10
) -> None:
    
    model.eval()
    images, labels = next(iter(test_loader))
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    
    _, axes = plt.subplots(
        2, num_images // 2, 
        figsize = (20, 8)
    )
    incorrect_idx = torch.where(preds != labels)[0]

    for i in range(num_images // 2):
        axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title(
            f'Pred: {preds[i]}\n \
            True: {labels[i]}'
        )
        axes[0, i].axis('off')
        
        if incorrect_idx.numel() > i:
            axes[1, i].imshow(images[incorrect_idx[i]].permute(1, 2, 0).cpu().numpy())
            axes[1, i].set_title(
                f'Pred: {preds[incorrect_idx[i]]}\n \
                True: {labels[incorrect_idx[i]]}'
            )
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

#%%

""" Training process """

def read_model_names(filename):
    with open(filename, 'r') as file:
        model_names = [line.strip() for line in file if line.strip()]
    return model_names

# load model name 
file_path : str = 'model_names.txt'
model_name: List[str] = read_model_names(file_path)

class CustomTrainingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_progress_bar = None
        self.val_progress_bar = None

    def on_train_epoch_start(
        self, 
        trainer, 
        pl_module
    ):
        tqdm.write(
            f"\n{Fore.CYAN}Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
        )
        self.train_progress_bar = tqdm(
            total = len(trainer.train_dataloader), 
            desc = f"{Fore.GREEN}Training", 
            leave = False
        )

    def on_train_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs, 
        batch, 
        batch_idx
    ):
        self.train_progress_bar.update(1)
        self.train_progress_bar.set_postfix(
            loss = f"{outputs['loss'].item():.4f}",
            acc = f"{outputs['train_acc'].item():.4f}",
        )

    def on_train_epoch_end(
        self, 
        trainer, 
        pl_module
    ):
        self.train_progress_bar.close()

    def on_validation_epoch_start(
        self, 
        trainer, 
        pl_module
    ):
        self.val_progress_bar = tqdm(
            total = len(trainer.val_dataloaders), 
            desc = f"{Fore.YELLOW}Validation", 
            leave = False
        )

    def on_validation_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs, 
        batch, 
        batch_idx, 
        dataloader_idx = 0
    ):
        self.val_progress_bar.update(1)

    def on_validation_epoch_end(
        self, 
        trainer, 
        pl_module
    ):
        
        self.val_progress_bar.close()
        metrics = trainer.callback_metrics

        tqdm.write(
            f"{Fore.MAGENTA}Val Loss: {metrics['val_loss']:.4f}, \
            Val Acc: {metrics['val_acc']:.4f}"
        )
        
def main() -> None:

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        description = 'Pytorch Lightning Classifier Model'
    )

    parser.add_argument(
        '--model', 
        type = str, 
        default = 'AlexNet',
        choices = model_name, 
        help = 'Model Architecture'
    )
    parser.add_argument(
        '--pretrained', 
        action = 'store_true',
        help = 'Model selection mode for training (default: False)'
    )
    parser.add_argument(
        '--batch-size', 
        type = int, 
        default = 12, 
        help = 'input batch size for training (default: 12)'
    )
    parser.add_argument(
        '--epochs', 
        type = int, 
        default = 10, 
        help = 'number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--lr', 
        type = float, 
        default = 0.001, 
        help = 'learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--no-cuda', 
        action = 'store_true', 
        default = False, 
        help = 'disables CUDA training'
    )
    parser.add_argument(
        '--seed', 
        type = int, 
        default = 1, 
        help = 'random seed (default: 1)'
    )
    
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406], 
            std = [0.229, 0.224, 0.225]
        ),
    ])
    
    train_dataset = datasets.CIFAR10(
        root = './data', 
        train = True, 
        download = True, 
        transform = transform
    )
    val_dataset = datasets.CIFAR10(
        root = './data', 
        train = False, 
        download = True, 
        transform = transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        shuffle = True, 
        num_workers = 4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size = args.batch_size, 
        shuffle = False, 
        num_workers = 4
    )

    # Model
    model = ImageClassifierModel(
        args.model, 
        num_classes = 10, 
        lr = args.lr, 
        pretrained = args.pretrained
    ).to(device)

    summary(
        model, 
        input_size = (1, 3, 224, 224)
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_loss',
        filename = '{epoch:02d}-{val_loss:.2f}',
        save_top_k = 3,
        mode = 'min'
    )
    early_stop_callback = EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        mode = 'min'
    )

    # Logger
    logger = TensorBoardLogger(
        "runs", 
        name = args.model
    )
    
    # Tạo banner đẹp mắt
    banner = pyfiglet.figlet_format("CIFAR-10 Training", font = "slant")
    tqdm.write(f"{Fore.CYAN}{banner}")

    # Thêm custom callback vào trainer
    custom_callback = CustomTrainingCallback()
    trainer = pl.Trainer(
        max_epochs = args.epochs,
        callbacks = [checkpoint_callback, early_stop_callback, custom_callback],
        logger = logger,
        log_every_n_steps = 10,
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
        devices = 1 if torch.cuda.is_available() else None,
        enable_progress_bar = False,  # Tắt thanh tiến trình mặc định
    )

    # Training
    tqdm.write(f"{Fore.GREEN} Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Testing
    tqdm.write(f"\n{Fore.YELLOW} Starting testing...")
    trainer.test(model, val_loader)

    # Show predictions
    tqdm.write(f"\n{Fore.MAGENTA} Showing predictions...")

    show_predictions(model, val_loader)

if __name__ == '__main__':
    main()

""" 
python train.py --model AlexNet --pretrained True --batch-size 15 --epochs 5 --lr 0.001
"""
# %%