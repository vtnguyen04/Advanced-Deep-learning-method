import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Union

def make_divisible(v: float, divisor: int, min_value = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1  # Corrected parameter name
    ):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                groups = groups,
                bias = False
            ),
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU6(inplace = True)  
        )

class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int
    ):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size = 1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride = stride, groups = hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return inputs + self.conv(inputs)
        else:
            return self.conv(inputs)

class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: List = None,
        round_nearest: int = 8
    ):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        in_channels = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        in_channels = make_divisible(
            in_channels * width_mult, 
            round_nearest
        )

        self.last_channel = make_divisible(
            last_channel * max(1.0, width_mult), 
            round_nearest
        )
        features = [ConvBNReLU(3, in_channels, stride = 2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(
                c * width_mult, 
                round_nearest
            )
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(in_channels, output_channel, stride, expand_ratio = t))
                in_channels = output_channel
        
        features.append(ConvBNReLU(in_channels, self.last_channel, kernel_size = 1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

def get_mobilenetv2(
    num_classes: int,
    pretrained: bool = False,
    width_mult: float = 1.0
):
    if not pretrained:
        return MobileNetV2(
            num_classes = num_classes,
            width_mult = width_mult
        )
        

    model = models.mobilenet_v2(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model
  

def get_mobilenet_v2(num_classes: int, pretrained: bool = False, width_mult: float = 1.0):
    return get_mobilenetv2(num_classes, pretrained, width_mult)