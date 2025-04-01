import torch
import torch.nn as nn
import timm
from typing import Type, List, Union

# class SeparableConv2d(nn.Module):
#     def __init__(self, 
#                  in_channels: int, 
#                  out_channels: int, 
#                  kernel_size: int = 3, 
#                  stride: int = 1, 
#                  padding: int = 1):
        
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_channels = in_channels, 
#                                    out_channels = in_channels, 
#                                    kernel_size = kernel_size,
#                                    stride = stride, 
#                                    padding = padding, 
#                                    groups = in_channels, 
#                                    bias = False)
#         self.pointwise = nn.Conv2d(in_channels = in_channels, 
#                                    out_channels = out_channels, 
#                                    kernel_size = 1, 
#                                    bias = False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace = True)

#     def forward(self, 
#                 x: torch.Tensor) -> torch.Tensor:
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

# class NormalCell(nn.Module):
#     def __init__(self, 
#                  in_channels: int, 
#                  out_channels: int):
        
#         super().__init__()
#         self.sep_conv1 = SeparableConv2d(in_channels = in_channels, 
#                                          out_channels = out_channels)
#         self.sep_conv2 = SeparableConv2d(in_channels = in_channels, 
#                                          out_channels = out_channels)

#     def forward(self, 
#                 x: torch.Tensor) -> torch.Tensor:
#         return self.sep_conv1(x) + self.sep_conv2(x)

# class ReductionCell(nn.Module):
#     def __init__(self, 
#                  in_channels: int, 
#                  out_channels: int):
        
#         super().__init__()
#         self.sep_conv1 = SeparableConv2d(in_channels = in_channels, 
#                                          out_channels = out_channels, 
#                                          stride = 2)
#         self.sep_conv2 = SeparableConv2d(in_channels = in_channels, 
#                                          out_channels = out_channels, 
#                                          stride = 2)
#         self.maxpool = nn.MaxPool2d(kernel_size = 3, 
#                                     stride = 2,  
#                                     padding = 1)
#         # Add a 1x1 convolution after maxpool to match channel dimensions
#         self.conv_maxpool = nn.Conv2d(in_channels = in_channels,
#                                      out_channels = out_channels,
#                                      kernel_size = 1,
#                                      bias = False)
#         self.bn = nn.BatchNorm2d(out_channels)

#     def forward(self, 
#                 x: torch.Tensor) -> torch.Tensor:
#         # Apply the 1x1 convolution to the maxpooled output to match channels
#         maxpool_out = self.maxpool(x)
#         maxpool_out = self.conv_maxpool(maxpool_out)
#         maxpool_out = self.bn(maxpool_out)
        
#         return self.sep_conv1(x) + self.sep_conv2(x) + maxpool_out

# class NasNet(nn.Module):
#     def __init__(self, 
#                  num_normal_cells: int = 6, 
#                  num_reduction_cells: int = 2, 
#                  num_classes: int = 1000):
        
#         super().__init__()

#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels = 3, 
#                       out_channels = 32, 
#                       kernel_size = 3, 
#                       stride = 2, 
#                       padding = 1, 
#                       bias = False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace = True)
#         )

#         self.layers = nn.ModuleList()
#         in_channels = 32
#         out_channels = 64

#         for i in range(num_reduction_cells):
#             for j in range(num_normal_cells):
#                 self.layers.append(NormalCell(in_channels, out_channels))
#                 in_channels = out_channels
            
#             if i < num_reduction_cells - 1:
#                 self.layers.append(ReductionCell(in_channels, out_channels * 2))
#                 in_channels = out_channels * 2
#                 out_channels *= 2

#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(in_channels, num_classes)

#     def forward(self, 
#                 x: torch.Tensor) -> torch.Tensor:
#         x = self.stem(x)

#         for layer in self.layers:
#             x = layer(x)

#         x = self.global_avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x

# def get_nasnet(model_name: str, 
#                num_classes: int, 
#                pretrained: bool = False):
#     timm_model_name = {
#         'nasnet_small': 'nasnetalarge',
#         'nasnet_large': 'nasnetamobile'
#     }.get(model_name, None)
    
#     if timm_model_name is None:
#         raise ValueError(f"Unsupported model_name: {model_name}")
    
#     if pretrained:
#         model = timm.create_model(timm_model_name, pretrained=True)
        
#         for param in model.parameters():
#             param.requires_grad = False
        
#         num_ftrs = model.last_linear.in_features
#         model.last_linear = nn.Linear(num_ftrs, num_classes)
#     else:
#         num_normal_cells = 6 if model_name == 'nasnet_small' else 18
#         model = NasNet(num_normal_cells = num_normal_cells, 
#                        num_reduction_cells = 2, 
#                        num_classes = num_classes)
    
#     return model

# def get_nasnet_small(num_classes: int, pretrained: bool = False):
#     return get_nasnet('nasnet_small', num_classes, pretrained)

# def get_nasnet_large(num_classes: int, pretrained: bool = False):
#     return get_nasnet('nasnet_large', num_classes, pretrained)

#%%

import torch
import torch.nn as nn
import timm
from typing import Type, List, Union
import torch.nn.functional as F

class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))  # Đã sửa thành F.relu
        x = F.relu(self.bn1(self.conv1(x)))  # Đã sửa thành F.relu
        x = F.relu(self.bn2(self.conv2(x)))  # Đã sửa thành F.relu
        return x

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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sep_conv_left = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels, eps=1e-3)
        )
        
        self.sep_conv_right = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-3)
        )

    def forward(self, x):
        return self.sep_conv_left(x) + self.sep_conv_right(x)

class ReductionCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        
        self.path1 = nn.Sequential(
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels//4, 1, bias=False),
            nn.BatchNorm2d(out_channels//4)
        )
        
        self.path2 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels//4, 1, bias=False),
            nn.BatchNorm2d(out_channels//4)
        )
        
        self.path3 = nn.Sequential(
            SeparableConv2d(in_channels, out_channels//2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels//2)
        )

    def forward(self, x):
        return torch.cat([self.path1(x), self.path2(x), self.path3(x)], dim=1)

class NASNet(nn.Module):
    def __init__(self, num_classes=1000, 
                 normal_cell_repeats=[3, 4, 5, 4, 3],
                 reduction_out_channels=[256, 512, 1024, 2048]):
        super().__init__()
        self.stem = Stem()
        
        self.cells = nn.ModuleList()
        in_channels = 256  # Output từ Stem
        
        # Khối Normal Cell đầu tiên
        for _ in range(normal_cell_repeats[0]):
            self.cells.append(NormalCell(in_channels, in_channels))
        
        # Các khối Reduction + Normal tiếp theo
        for i in range(len(reduction_out_channels)):
            # Thêm Reduction Cell
            self.cells.append(ReductionCell(in_channels, reduction_out_channels[i]))
            in_channels = reduction_out_channels[i]
            
            # Thêm Normal Cells
            for _ in range(normal_cell_repeats[i+1]):
                self.cells.append(NormalCell(in_channels, in_channels))
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_channels, num_classes)

def get_nasnet(model_name, num_classes=1000, pretrained=False):
    configs = {
        'nasnet_large': {
            'normal_cell_repeats': [3, 4, 5, 4, 3],
            'reduction_out_channels': [512, 1024, 2048, 4096]
        },
        'nasnet_mobile': {
            'normal_cell_repeats': [2, 3, 4, 3, 2],
            'reduction_out_channels': [256, 512, 1024, 2048]
        }
    }
    
    if pretrained:
        model = timm.create_model('nasnetalarge' if 'large' in model_name else 'nasnetamobile', 
                                 pretrained=True)
        model.reset_classifier(num_classes)
    else:
        config = configs[model_name]
        model = NASNet(
            num_classes=num_classes,
            normal_cell_repeats=config['normal_cell_repeats'],
            reduction_out_channels=config['reduction_out_channels']
        )
    
    return model

def get_nasnet_small(num_classes: int, pretrained: bool = False):
    return get_nasnet('nasnet_mobile', num_classes, pretrained)

def get_nasnet_large(num_classes: int, pretrained: bool = False):
    return get_nasnet('nasnet_large', num_classes, pretrained)