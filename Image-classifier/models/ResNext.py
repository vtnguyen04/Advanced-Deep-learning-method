import torch
import torch.nn as nn
import torchvision.models as models

class ResNeXtBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, cardinality, bottleneck_width, stride=1):
        super(ResNeXtBlock, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_channels, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, group_width * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(group_width * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != group_width * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, group_width * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(group_width * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality, bottleneck_width, num_classes):
        super(ResNeXt, self).__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[3], stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, self.cardinality, self.bottleneck_width, stride))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, self.cardinality, self.bottleneck_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_resnext50(num_classes, pretrained=False):
    if pretrained:
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = ResNeXt(ResNeXtBlock, [3, 4, 6, 3], 32, 4, num_classes)
    return model