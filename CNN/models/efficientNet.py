import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Union

class MBConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1

        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        
        if self.expand:
            self.expand_conv = self._make_layer(
                in_channels = in_channels, 
                out_channels = hidden_dim, 
                kernel_size = 1
            )

        self.depthwise_conv = self._make_layer(
            in_channels = hidden_dim, 
            out_channels = hidden_dim, 
            kernel_size = kernel_size, 
            stride = stride, 
            groups = hidden_dim
        )
        self.se = SqueezeExcitation(
            in_channels = hidden_dim, 
            squeeze_channels = int(in_channels * se_ratio)
        )
        self.project_conv = self._make_layer(
            in_channels = hidden_dim, 
            out_channels = out_channels, 
            kernel_size = 1, 
            activation = False
        )

    def _make_layer(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        groups: int = 1,
        activation: bool = True
    ) -> nn.Sequential:
        layers = [
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = kernel_size, 
                stride = stride, 
                padding = kernel_size // 2, 
                groups = groups, 
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace = True))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs

        if self.expand:
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)

        if self.use_residual:
            return x + inputs
        else:
            return x

class SqueezeExcitation(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        squeeze_channels: int
    ):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, 1),
            nn.SiLU(inplace = True),
            nn.Conv2d(squeeze_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale

class EfficientNet(nn.Module):
    def __init__(
        self, 
        width_coefficient: float, 
        depth_coefficient: float, 
        num_classes: int = 1000
    ):
        super().__init__()
        
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        expand_ratios = [1, 6, 6, 6, 6, 6, 6]

        channels = [int(c * width_coefficient) for c in channels]
        repeats = [int(r * depth_coefficient) for r in repeats]

        self.stem = self._make_layer(
            in_channels = 3, 
            out_channels = channels[0], 
            kernel_size = 3, 
            stride = 2
        )

        self.blocks = nn.Sequential()
        in_channels = channels[0]
        for i in range(7):
            for j in range(repeats[i]):
                stride = strides[i] if j == 0 else 1
                self.blocks.add_module(f"block_{i}_{j}", MBConvBlock(
                    in_channels = in_channels, 
                    out_channels = channels[i+1], 
                    kernel_size = kernel_sizes[i], 
                    stride = stride, 
                    expand_ratio = expand_ratios[i]
                ))
                in_channels = channels[i+1]

        self.head = self._make_layer(
            in_channels = in_channels, 
            out_channels = channels[-1], 
            kernel_size = 1
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def _make_layer(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = kernel_size, 
                stride = stride, 
                padding = kernel_size // 2, 
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace = True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def efficientnet_b0(num_classes: int = 1000):
    return EfficientNet(1.0, 1.0, num_classes)

def efficientnet_b1(num_classes: int = 1000):
    return EfficientNet(1.0, 1.1, num_classes)

def efficientnet_b2(num_classes: int = 1000):
    return EfficientNet(1.1, 1.2, num_classes)

def get_efficientnet(
    model_name: str, 
    num_classes: int, 
    pretrained: bool = False
):
    model_func = globals()[model_name]
    if pretrained:
        model = getattr(models, model_name)(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    else:
        model = model_func(num_classes = num_classes)
    return model

def get_efficientnet_b0(num_classes: int, pretrained: bool = False):
    return get_efficientnet('efficientnet_b0', num_classes, pretrained)

def get_efficientnet_b1(num_classes: int, pretrained: bool = False):
    return get_efficientnet('efficientnet_b1', num_classes, pretrained)

def get_efficientnet_b2(num_classes: int, pretrained: bool = False):
    return get_efficientnet('efficientnet_b2', num_classes, pretrained)