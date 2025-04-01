import torch
import torch.nn as nn
import torchvision.models as models
import math
from typing import List, Union, Optional, Callable

def round_filters(filters, width_coefficient, divisor=8):
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))

class SqueezeExcitation(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        squeeze_channels: int,
        activation: Callable = nn.SiLU
    ):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, 1),
            activation(inplace = True),
            nn.Conv2d(squeeze_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale

class MBConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
        activation: Callable = nn.SiLU
    ):
        super().__init__()
        self.stride = stride
        self.drop_path_rate = drop_path_rate
        self.use_residual = in_channels == out_channels and stride == 1

        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        
        if self.expand:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels, 
                    out_channels = hidden_dim, 
                    kernel_size = 1, 
                    bias = False
                ),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace = True)
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = hidden_dim, 
                out_channels = hidden_dim, 
                kernel_size = kernel_size, 
                stride = stride, 
                padding = kernel_size//2, 
                groups = hidden_dim, 
                bias = False
            ),
            nn.BatchNorm2d(hidden_dim),
            activation(inplace = True)
        )

        squeeze_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(
            in_channels = hidden_dim, 
            squeeze_channels = squeeze_channels,
            activation = activation
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = hidden_dim, 
                out_channels = out_channels, 
                kernel_size = 1, 
                bias = False
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs

        if self.expand:
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)

        if self.use_residual:
            x = self.drop_path(x) + inputs
        
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype = x.dtype, device = x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class EfficientNet(nn.Module):
    def __init__(
        self, 
        width_coefficient: float, 
        depth_coefficient: float, 
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
        num_classes: int = 1000,
        activation: Callable = nn.SiLU
    ):
        super().__init__()
        
        self.base_model_config = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        
        out_channels = round_filters(32, width_coefficient)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels = 3, 
                out_channels = out_channels, 
                kernel_size = 3, 
                stride = 2, 
                padding = 1, 
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            activation(inplace = True)
        )
        
        self.blocks = nn.ModuleList([])
        
        in_channels = out_channels
        
        total_blocks = sum(round_repeats(config[2], depth_coefficient) for config in self.base_model_config)
        block_idx = 0
        
        for config in self.base_model_config:
            expand_ratio, channels, repeats, stride, kernel_size = config
            
            out_channels = round_filters(channels, width_coefficient)
            repeats = round_repeats(repeats, depth_coefficient)
            
            for i in range(repeats):
                block_stride = stride if i == 0 else 1
                drop_path_rate = drop_connect_rate * block_idx / total_blocks
                
                block = MBConvBlock(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = block_stride,
                    expand_ratio = expand_ratio,
                    se_ratio = 0.25,
                    drop_path_rate = drop_path_rate,
                    activation = activation
                )
                
                self.blocks.append(block)
                
                in_channels = out_channels
                block_idx += 1
        
        last_channels = round_filters(1280, width_coefficient)
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = last_channels, 
                kernel_size = 1, 
                bias = False
            ),
            nn.BatchNorm2d(last_channels),
            activation(inplace = True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(last_channels, num_classes)

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

def efficientnet_b0(num_classes: int = 1000, dropout_rate: float = 0.2):
    return EfficientNet(
        width_coefficient = 1.0,
        depth_coefficient = 1.0,
        dropout_rate = dropout_rate,
        num_classes = num_classes
    )

def efficientnet_b1(num_classes: int = 1000, dropout_rate: float = 0.2):
    return EfficientNet(
        width_coefficient = 1.0,
        depth_coefficient = 1.1,
        dropout_rate = dropout_rate,
        num_classes = num_classes
    )

def efficientnet_b2(num_classes: int = 1000, dropout_rate: float = 0.3):
    return EfficientNet(
        width_coefficient = 1.1,
        depth_coefficient = 1.2,
        dropout_rate = dropout_rate,
        num_classes = num_classes
    )

def efficientnet_b3(num_classes: int = 1000, dropout_rate: float = 0.3):
    return EfficientNet(
        width_coefficient = 1.2,
        depth_coefficient = 1.4,
        dropout_rate = dropout_rate,
        num_classes = num_classes
    )

def get_efficientnet(
    model_name: str, 
    num_classes: int, 
    pretrained: bool = False
):
    model_func = globals()[model_name]
    
    if pretrained:
        try:
            model = getattr(models, model_name)(pretrained = True)
            for param in model.parameters():
                param.requires_grad = False
                
            if hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Linear):
                    num_ftrs = model.classifier.in_features
                    model.classifier = nn.Linear(num_ftrs, num_classes)
                else:
                    num_ftrs = model.classifier[1].in_features
                    model.classifier = nn.Sequential(
                        nn.Dropout(p = 0.3, inplace = True),
                        nn.Linear(num_ftrs, num_classes),
                    )
                    
        except (AttributeError, ImportError):
            print(f"Pretrained {model_name} not found in torchvision, creating from scratch")
            model = model_func(num_classes = num_classes)
    else:
        model = model_func(num_classes = num_classes)
        
    return model

def get_efficientnet_b0(num_classes: int, pretrained: bool = False):
    return get_efficientnet('efficientnet_b0', num_classes, pretrained)

def get_efficientnet_b1(num_classes: int, pretrained: bool = False):
    return get_efficientnet('efficientnet_b1', num_classes, pretrained)

def get_efficientnet_b2(num_classes: int, pretrained: bool = False):
    return get_efficientnet('efficientnet_b2', num_classes, pretrained)

def get_efficientnet_b3(num_classes: int, pretrained: bool = False):
    return get_efficientnet('efficientnet_b3', num_classes, pretrained)