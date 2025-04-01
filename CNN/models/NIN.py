import torch
import torch.nn as nn

""" lần đầu sử dụng 1x1 conv, adaptive-avarage-pooling toàn bộ thay vì flatten """

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0
    ):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                bias = False
            ),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(inplace = True)  
        )

class MLPConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(MLPConv, self).__init__(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                bias = False
            ),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(inplace = True)  
        )

class NetworkInNetwork(nn.Module):
    def __init__(
        self, 
        num_classes: int = 10
    ):
        super().__init__()

        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBNReLU(
                in_channels = 3,
                out_channels = 192,
                kernel_size = 5, 
                stride = 1, 
                padding = 2
            ),
            MLPConv(
                in_channels = 192, 
                out_channels = 160 
            ),
            MLPConv(
                in_channels = 160, 
                out_channels = 96 
            ),
            nn.MaxPool2d(
                kernel_size = 3, 
                stride = 2, 
                padding = 1
            ),
            nn.Dropout(0.5)
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBNReLU(
                in_channels = 96, 
                out_channels = 192, 
                kernel_size = 5, 
                stride = 1, 
                padding = 2
            ),
            MLPConv(
                in_channels = 192, 
                out_channels = 192
            ),
            MLPConv(
                in_channels = 192, 
                out_channels = 192
            ),
            nn.MaxPool2d(
                kernel_size = 3, 
                stride = 2, 
                padding = 1
            ),
            nn.Dropout(0.5)
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBNReLU(
                in_channels = 192, 
                out_channels = 192, 
                kernel_size = 3, 
                stride = 1, 
                padding = 1
            ),
            MLPConv(
                in_channels = 192, 
                out_channels = 192
            ),
            MLPConv(
                in_channels = 192, 
                out_channels = num_classes
            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.stage1(inputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = outputs.view(outputs.size(0), -1)  
        return outputs

def nin(
    num_classes: int = 10
) -> NetworkInNetwork:
    return NetworkInNetwork(num_classes = num_classes)

def get_nin(
    num_classes: int, 
    pretrained: bool = False
) -> nn.Module:
    
    if pretrained:
        raise NotImplementedError("NIN model does not support pretrained weights.")
    else:
        return nin(num_classes=num_classes)

