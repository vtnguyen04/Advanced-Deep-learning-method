import torch
import torch.nn as nn
import torchvision.models as models
from typing import Type, List, Union

class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, 
            bn_size * growth_rate, 
            kernel_size = 1, 
            bias = False
        )
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate, 
            growth_rate, 
            kernel_size = 3, 
            padding = 1, 
            bias = False
        )
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        
        outputs = self.bn1(inputs)
        outputs = self.relu(outputs)
        outputs = self.conv1(outputs)
        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout(outputs)

        return torch.cat([inputs, outputs], 1)

class DenseBlock(nn.Module):
    def __init__(
        self, 
        num_layers: int, 
        in_channels: int, 
        bn_size: int, 
        growth_rate: int, 
        dropout_rate: float
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                dropout_rate
            ) for i in range(num_layers)
        ])

    def forward(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

class Transition(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size = 1, 
            bias = False
        )
        self.relu = nn.ReLU(inplace = True)
        self.pool = nn.AvgPool2d(
            kernel_size = 2, 
            stride = 2
        )

    def forward(
            self, 
            inputs: torch.Tensor
        ) -> torch.Tensor:

        outputs = self.bn(inputs)
        outputs = self.relu(outputs)
        outputs = self.conv(outputs)
        outputs = self.pool(outputs)
        return outputs

class DenseNet(nn.Module):
    def __init__(
        self, 
        block_config: List[int],
        in_channels = 3,
        growth_rate: int = 32, 
        num_init_channels: int = 64, 
        bn_size: int = 4, 
        dropout_rate: float = 0,
        num_classes: int = 10
    ):
        super().__init__()

        # First convolution
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                num_init_channels, 
                kernel_size = 7, 
                stride = 2, 
                padding = 3, 
                bias = False
            ),
            nn.BatchNorm2d(num_init_channels),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(
                kernel_size = 3, 
                stride = 2, 
                padding = 1
            ),
        )

        
        # Dense blocks
        num_channels = num_init_channels
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers = num_layers,
                in_channels = num_channels,
                bn_size = bn_size,
                growth_rate = growth_rate,
                dropout_rate = dropout_rate
            )
            self.features.add_module(
                f'denseblock{i + 1}', 
                block
            )
            num_channels = num_channels + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(
                    in_channels = num_channels, 
                    out_channels = num_channels // 2
                )
                self.features.add_module(
                    f'transition{i + 1}', 
                    trans
                )
                num_channels = num_channels // 2

        # Final batch norm
        self.features.add_module(
            'norm5', 
            nn.BatchNorm2d(num_channels)
        )

        # Linear layer
        self.classifier = nn.Linear(num_channels, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace = True)

    def forward(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        
        features = self.features(inputs)
        outputs = self.relu(features)
        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)
        outputs = self.classifier(outputs)
        return outputs

def densenet121(
    num_classes: int
):
    return DenseNet(
        growth_rate = 32, 
        block_config = (6, 12, 24, 16), 
        num_classes = num_classes
    )

def densenet169(
    num_classes: int
):
    return DenseNet(
        growth_rate = 32, 
        block_config = (6, 12, 32, 32), 
        num_classes = num_classes
    )

def densenet201(
    num_classes: int
):
    return DenseNet(
        growth_rate = 32, 
        block_config = (6, 12, 48, 32), 
        num_classes = num_classes
    )

def get_densenet(
    model_name: str, 
    num_classes: int, 
    pretrained: bool = False
):
    
    model_func = globals()[model_name]
    if pretrained:
        model = getattr(models, model_name)(pretrained = True)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        model = model_func(num_classes = num_classes)
    return model

def get_densenet121(
    num_classes: int, 
    pretrained: bool = False
):
    return get_densenet(
        'densenet121', 
        num_classes, 
        pretrained
    )

def get_densenet169(
    num_classes: int, 
    pretrained: bool = False
):
    return get_densenet(
        'densenet169', 
        num_classes, 
        pretrained
    )

def get_densenet201(
    num_classes: int, 
    pretrained: bool = False
):
    return get_densenet(
        'densenet201', 
        num_classes, 
        pretrained
    )