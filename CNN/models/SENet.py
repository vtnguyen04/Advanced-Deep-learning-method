import torch
import torch.nn as nn
import timm
from typing import Type, List, Union, Optional

class SEModule(nn.Module):
    def __init__(self,
                 channels: int,
                 reduction_ratio: int):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, 
                             channels // reduction_ratio)
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(channels // reduction_ratio, 
                             channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        module_input = x
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x).view(x.size(0), x.size(1), 1, 1)
        return module_input * x

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Type[nn.Module]] = None,
                 reduction_ratio: int = 16):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, 
                               width, 
                               kernel_size = 1, 
                               bias = False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width,
                               width,
                               kernel_size = 3,
                               stride = stride,
                               padding = dilation,
                               groups = groups,
                               bias = False,
                               dilation = dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, 
                               planes * self.expansion, 
                               kernel_size = 1, 
                               bias = False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.se_module = SEModule(planes * self.expansion, 
                                  reduction_ratio)
        self.downsample = downsample

        self.stride = stride

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se_module(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SENet(nn.Module):
    def __init__(self,
                 block: Type[Bottleneck],
                 layers: List[int],
                 groups: int = 1,
                 width_per_group: int = 64,
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 norm_layer: Optional[Type[nn.Module]] = None,
                 reduction_ratio: int = 16):
        super(SENet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size = 7,
                               stride = 2,
                               padding = 3,
                               bias = False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, 
                                    stride = 2, 
                                    padding = 1)
        self.layer1 = self._make_layer(block, 
                                       64, 
                                       layers[0], 
                                       reduction_ratio = reduction_ratio)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride = 2,
                                       reduction_ratio = reduction_ratio)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride = 2,
                                       reduction_ratio = reduction_ratio)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride = 2,
                                       reduction_ratio = reduction_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode = 'fan_out', 
                                        nonlinearity = 'relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self,
                    block: Type[Bottleneck],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    reduction_ratio: int = 16) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size = 1,
                          stride = stride,
                          bias = False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes,
                            planes,
                            stride,
                            downsample,
                            self.groups,
                            self.base_width,
                            self.dilation,
                            norm_layer,
                            reduction_ratio))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes,
                                planes,
                                groups = self.groups,
                                base_width = self.base_width,
                                dilation = self.dilation,
                                norm_layer = norm_layer,
                                reduction_ratio = reduction_ratio))

        return nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def se_resnet50(num_classes: int = 1000,
                pretrained: bool = False) -> nn.Module:
    if pretrained:
        model = timm.create_model('seresnet50', 
                                  pretrained = True)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, 
                                 num_classes)
    else:
        model = SENet(Bottleneck, 
                      [3, 4, 6, 3], 
                      num_classes = num_classes)
    return model

def se_resnet101(num_classes: int = 1000,
                 pretrained: bool = False) -> nn.Module:
    if pretrained:
        model = timm.create_model('seresnet101', 
                                  pretrained = True)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, 
                                 num_classes)
    else:
        model = SENet(Bottleneck, 
                      [3, 4, 23, 3],
                      num_classes = num_classes)
    return model

def se_resnet152(num_classes: int = 1000,
                 pretrained: bool = False) -> nn.Module:
    if pretrained:
        model = timm.create_model('seresnet152', pretrained = True)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = SENet(Bottleneck, [3, 8, 36, 3], num_classes = num_classes)
    return model