import torch
import torch.nn as nn
import torchvision.models as models
from typing import Type, List, Union

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        downsample: Union[nn.Module, None] = None
    ):

        super().__init__()
        self.conv1 = self._make_layer(
            in_channels, 
            out_channels, 
            kernel_size = 3, 
            stride = stride, 
            padding = 1
        )
        self.conv2 = self._make_layer(
            out_channels, 
            out_channels * self.expansion, 
            kernel_size = 3, 
            padding = 1
        )
        
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample

    def _make_layer(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0
    ) -> nn.Sequential:
        
        return nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = kernel_size, 
                padding = padding, 
                stride = stride, 
                bias = False
            ),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        
        identity = inputs

        outputs = self.conv1(inputs)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        outputs += identity
        outputs = self.relu(outputs)
        return outputs

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        downsample: Union[nn.Module, None] = None
    ):
        
        super().__init__()
        self.conv1 = self._make_layer(
            in_channels, 
            out_channels, 
            kernel_size = 1)
        
        self.conv2 = self._make_layer(
            out_channels, 
            out_channels, 
            kernel_size = 3, 
            stride = stride, 
            padding = 1
        )

        self.conv3 = self._make_layer(
            out_channels, 
            out_channels * self.expansion, 
            kernel_size = 1
        )
        
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
    
    def _make_layer(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0
    ) -> nn.Sequential:
        
        return nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = kernel_size, 
                padding = padding, 
                stride = stride, 
                bias = False
            ),
            nn.BatchNorm2d(out_channels)
        )   
    
    def forward(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        identity = inputs

        outputs = self.conv1(inputs)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv3(outputs)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        outputs += identity
        outputs = self.relu(outputs)
        return outputs

class ResNet(nn.Module):
    def __init__(
        self, 
        block: Type[nn.Module], 
        blocks: List[int], 
        num_classes: int = 1000
    ):
        
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3, 64, 
            kernel_size = 7,
            stride = 2,
            padding = 3,
            bias = False
        )
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(
            kernel_size = 3, 
            stride = 2, 
            padding = 1
        )
        
        self.stage1 = self._make_layer(
            block = block, 
            out_channels = 64, 
            blocks = blocks[0]
        )
        self.stage2 = self._make_layer(
            block = block, 
            out_channels = 128, 
            blocks = blocks[1], 
            stride = 2
        )

        self.stage3 = self._make_layer(
            block = block, 
            out_channels = 256, 
            blocks = blocks[2], 
            stride = 2
        )

        self.stage4 = self._make_layer(
            block = block, 
            out_channels = 512, 
            blocks = blocks[3], 
            stride = 2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            512 * block.expansion, 
            num_classes
        )

    def _make_layer(
        self, 
        block: Type[nn.Module], 
        out_channels: int, 
        blocks: int, 
        stride: int = 1
    ) -> nn.Sequential:

        downsample = None

        # This condition is unnecessary as it always holds true for each stage.
        if stride != 1 or self.in_channels != out_channels * block.expansion: 
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels * block.expansion, 
                    kernel_size = 1, 
                    stride = stride, 
                    bias = False
                ),
                nn.BatchNorm2d(
                    out_channels * block.expansion
                ),
            )

        layers = []
        layers.append(
            block(
                in_channels = self.in_channels, 
                out_channels = out_channels, 
                stride = stride, 
                downsample = downsample
            )
        ) # dowsample at first block
        
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels, 
                    out_channels
                )
            ) # keep dimension

        return nn.Sequential(*layers)

    def forward(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        
        outputs = self.conv1(inputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)

        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)

        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)
        outputs = self.fc(outputs)
        return outputs


def resnet18(
    num_classes: int
):
    
    return ResNet(
        BasicBlock, 
        blocks = [2, 2, 2, 2], 
        num_classes = num_classes
    )

def resnet50(
    num_classes: int
):
    
    return ResNet(
        Bottleneck, 
        blocks = [3, 4, 6, 3], 
        num_classes = num_classes
    )

def resnet101(
    num_classes: int
):
    
    return ResNet(
        Bottleneck, 
        blocks = [3, 4, 23, 3], 
        num_classes = num_classes
    )

def get_resnet(
    model_name: str, 
    num_classes: int, 
    pretrained: bool = False
):
    
    model_func = globals()[model_name]
    if pretrained:
        model = getattr(
            models, 
            model_name
        )(pretrained = True)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        # model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        model = model_func(
            num_classes = num_classes
        )
    return model


def get_resnet50(
    num_classes: int, 
    pretrained: bool = False
):
    
    return get_resnet(
        'resnet50', 
        num_classes, 
        pretrained
    )

def get_resnet18(
    num_classes: int, 
    pretrained: bool = False
):
    
    return get_resnet(
        'resnet18', 
        num_classes, 
        pretrained
    )


def get_resnet101(
    num_classes: int, 
    pretrained: bool = False
):
    
    return get_resnet(
        'resnet101', 
        num_classes, 
        pretrained
    )
