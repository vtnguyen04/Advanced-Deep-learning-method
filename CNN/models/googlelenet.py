import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple

class InceptionModule(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        n1x1: int, 
        n3x3red: int, 
        n3x3: int, 
        n5x5red: int, 
        n5x5: int, 
        pool_proj: int
    ):
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = n1x1, 
                kernel_size = 1
            )
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = n3x3red, 
                kernel_size = 1
            ),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels = n3x3red, 
                out_channels = n3x3, 
                kernel_size = 3, 
                padding = 1
            )
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = n5x5red, 
                kernel_size = 1
            ),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels = n5x5red, 
                out_channels = n5x5, 
                kernel_size = 5, 
                padding = 2
            )
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size = 3, 
                stride = 1, 
                padding = 1
            ),
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = pool_proj, 
                kernel_size = 1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class GoogLeNet(nn.Module):
    def __init__(
        self, 
        num_classes: int = 1000
    ):
        super(GoogLeNet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(
                in_channels = 3, 
                out_channels = 64, 
                kernel_size = 7, 
                stride = 2, 
                padding = 3
            ),
            nn.MaxPool2d(
                kernel_size = 3, 
                stride = 2, 
                padding = 1
            ),
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 1
            ),
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 192, 
                kernel_size = 3, 
                padding = 1
            ),
            nn.MaxPool2d(
                kernel_size = 3, 
                stride = 2, 
                padding = 1
            )
        )

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(
            kernel_size = 3, 
            stride = 2, 
            padding = 1
        )

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(
            kernel_size = 3, 
            stride = 2, 
            padding = 1
        )

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_layers(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def googlenet(num_classes: int):
    return GoogLeNet(num_classes = num_classes)

def get_googlenet(
    num_classes: int, 
    pretrained: bool = False
):
    if pretrained:
        model = models.googlenet(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    else:
        model = googlenet(num_classes = num_classes)
    return model