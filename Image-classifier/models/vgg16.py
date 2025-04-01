import torch
import torch.nn as nn
import torchvision.models as models

""" sử dụng 3x3 xuyên suốt + Max Pooling 2x2 , stride = 2 tạo mạng sâu hơn """
class VGG16(nn.Module):

    def __init__(
        self, 
        num_classes: int = 10
    ):
        super().__init__()
        self.features = nn.Sequential(
            self._make_block(3, 64, 2),
            self._make_block(64, 128, 2),
            self._make_block(128, 256, 3),
            self._make_block(256, 512, 3),
            self._make_block(512, 512, 3)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_block(
        self, 
        in_channels: int,
        out_channels: int, 
        conv_num: int
    ):

        layers = []
        for _ in range(conv_num):
            layers.extend([
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size = 3, 
                    padding = 1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace = True)
            ])
            in_channels = out_channels

        layers.append(nn.MaxPool2d(
            kernel_size = 2, 
            stride = 2
        ))
        return nn.Sequential(*layers)

    def forward(
            self,
            inputs: torch.Tensor
    ) -> torch.Tensor:

        inputs = self.features(inputs) 
        inputs = self.avgpool(inputs)
        inputs = torch.flatten(inputs, 1)
        inputs = self.classifier(inputs)
        return inputs

def get_vgg16(
        num_classes, 
        pretrained = False
):
    
    if pretrained:
        model = models.vgg16(pretrained = True)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier[0].in_features

        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        # num_ftrs = model.classifier[3].in_features
        # model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    else:
        model = VGG16(num_classes)
    return model