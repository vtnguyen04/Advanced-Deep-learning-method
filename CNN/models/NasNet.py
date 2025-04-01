import torch
import torch.nn as nn
import timm
from typing import Type, List, Union

class SeparableConv2d(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 stride: int = 1, 
                 padding: int = 1):
        
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels = in_channels, 
                                   out_channels = in_channels, 
                                   kernel_size = kernel_size,
                                   stride = stride, 
                                   padding = padding, 
                                   groups = in_channels, 
                                   bias = False)
        self.pointwise = nn.Conv2d(in_channels = in_channels, 
                                   out_channels = out_channels, 
                                   kernel_size = 1, 
                                   bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class NormalCell(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int):
        
        super().__init__()
        self.sep_conv1 = SeparableConv2d(in_channels = in_channels, 
                                         out_channels = out_channels)
        self.sep_conv2 = SeparableConv2d(in_channels = in_channels, 
                                         out_channels = out_channels)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        return self.sep_conv1(x) + self.sep_conv2(x)

class ReductionCell(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int):
        
        super().__init__()
        self.sep_conv1 = SeparableConv2d(in_channels = in_channels, 
                                         out_channels = out_channels, 
                                         stride = 2)
        self.sep_conv2 = SeparableConv2d(in_channels = in_channels, 
                                         out_channels = out_channels, 
                                         stride = 2)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, 
                                    stride = 2, 
                                    padding = 1)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        return self.sep_conv1(x) + self.sep_conv2(x) + self.maxpool(x)

class NasNet(nn.Module):
    def __init__(self, 
                 num_normal_cells: int = 6, 
                 num_reduction_cells: int = 2, 
                 num_classes: int = 1000):
        
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 3, 
                      out_channels = 32, 
                      kernel_size = 3, 
                      stride = 2, 
                      padding = 1, 
                      bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)
        )

        self.layers = nn.ModuleList()
        in_channels = 32
        out_channels = 64

        for i in range(num_reduction_cells):
            for j in range(num_normal_cells):
                self.layers.append(NormalCell(in_channels, out_channels))
                in_channels = out_channels
            
            if i < num_reduction_cells - 1:
                self.layers.append(ReductionCell(in_channels, out_channels * 2))
                in_channels = out_channels * 2
                out_channels *= 2

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for layer in self.layers:
            x = layer(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def nasnet_small(num_classes: int, 
                 pretrained: bool = False):
    
    if pretrained:
        model = timm.create_model('nasnetalarge', 
                                  pretrained = True)
        
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace the last fully connected layer
        num_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(num_ftrs, num_classes)
    else:
        model = NasNet(num_normal_cells = 6, 
                       num_reduction_cells = 2, 
                       num_classes = num_classes)
    
    return model

def nasnet_large(num_classes: int, 
                 pretrained: bool = False):
    
    if pretrained:
        model = timm.create_model('nasnetamobile', 
                                  pretrained = True)
        
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace the last fully connected layer
        num_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(num_ftrs, num_classes)
    else:
        model = NasNet(num_normal_cells = 18, 
                       num_reduction_cells = 2, 
                       num_classes = num_classes)
    
    return model

def get_nasnet(model_name: str, 
               num_classes: int, 
               pretrained: bool = False):
    
    model_func = globals()[model_name]
    model = model_func(num_classes = num_classes, 
                       pretrained = pretrained)
    return model

def get_nasnet_small(num_classes: int, 
                     pretrained: bool = False):
    
    return get_nasnet('nasnet_small', 
                      num_classes, 
                      pretrained)

def get_nasnet_large(num_classes: int, 
                     pretrained: bool = False):
    
    return get_nasnet('nasnet_large', 
                      num_classes, 
                      pretrained)