import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Union

class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = 3,
            stride = stride,
            padding = padding,
            groups = in_channels,
            bias = False
        )
        self.pointwise = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.depthwise(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.pointwise(outputs)
        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)
        return outputs

class MobileNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_multiplier: float = 1.0 
    ):
        super().__init__()
        
        def conv_bn(
            in_channels: int,
            out_channels: int,
            stride: int = 1
        ):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 3,
                    stride = stride,
                    padding = 1,
                    bias = False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace = True)
            )

        def conv_dw(
            in_channels: int, 
            out_channels: int, 
            stride: int = 1
        ):
            return DepthwiseSeparableConv(
                in_channels = in_channels,
                out_channels = out_channels,
                stride = stride
            )

        self.model = nn.Sequential(
            conv_bn(3, int(32 * width_multiplier), 2),
            conv_dw(int(32 * width_multiplier), int(64 * width_multiplier), 1),
            conv_dw(int(64 * width_multiplier), int(128 * width_multiplier), 2),
            conv_dw(int(128 * width_multiplier), int(128 * width_multiplier), 1),
            conv_dw(int(128 * width_multiplier), int(256 * width_multiplier), 2),
            conv_dw(int(256 * width_multiplier), int(256 * width_multiplier), 1),
            conv_dw(int(256 * width_multiplier), int(512 * width_multiplier), 2),
            *[conv_dw(int(512 * width_multiplier), int(512 * width_multiplier), 1) for _ in range(5)],
            conv_dw(int(512 * width_multiplier), int(1024 * width_multiplier), 2),
            conv_dw(int(1024 * width_multiplier), int(1024 * width_multiplier), 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(int(1024 * width_multiplier), num_classes)

    def forward(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.fc(outputs)
        return outputs

def mobilenet_v1(
    num_classes: int = 1000,
    width_multiplier: float = 1.0
):
    return MobileNet(
        num_classes = num_classes,
        width_multiplier = width_multiplier
    )

def get_mobilenet(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    width_multiplier: float = 1.0
):
    if pretrained:
        model = mobilenet_v1(
            num_classes = num_classes,
            width_multiplier = width_multiplier
        )
    else:
        model = mobilenet_v1(
            num_classes = num_classes,
            width_multiplier = width_multiplier
        )
    return model

def get_mobilenet_v1(
    num_classes: int,
    pretrained: bool = False,
    width_multiplier: float = 1.0
):
    return get_mobilenet(
        'mobilenet',
        num_classes,
        pretrained,
        width_multiplier
    )