import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(
        self, 
        num_classes: int = 1000
    ):
        super().__init__()

        self.features= nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 64,
                kernel_size = 11, 
                stride = 4, 
                padding = 2
            ),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(
                kernel_size = 3, 
                stride = 2
            ),
            nn.Conv2d(
                in_channels = 64,
                out_channels = 192,
                kernel_size = 5, 
                padding = 2
            ),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(
                kernel_size = 3, 
                stride = 2
            ),
            nn.Conv2d(
                in_channels = 192,
                out_channels = 384,
                kernel_size = 3, 
                padding = 1
            ),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels = 384,
                out_channels = 256,
                kernel_size = 3, 
                padding = 1
            ),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels = 256,
                out_channels = 256,
                kernel_size = 3, 
                padding = 1
            ),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(
                kernel_size = 3, 
                stride = 2
            ),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, num_classes),
        )

    def forward(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        
        inputs = self.features(inputs)
        inputs = self.avgpool(inputs)
        inputs = torch.flatten(inputs, 1)
        inputs = self.classifier(inputs)
        return inputs

def get_alexnet(
    num_classes: int, 
    pretrained: bool = False
):
    
    if pretrained: 
        model = models.alexnet(pretrained = True)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, num_classes)
        )

        # model.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        model = AlexNet(num_classes)
    return model