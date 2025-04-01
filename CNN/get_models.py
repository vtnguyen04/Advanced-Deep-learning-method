from models.AlexNet import get_alexnet
from models.NIN import get_nin
from models.mobilenet import get_mobilenet_v1
from models.resnet import get_resnet50, get_resnet101, get_resnet18
from models.vgg16 import get_vgg16
from models.googlelenet import get_googlenet

def get_model(
        model_name: str,
        num_classes: int,
        pretrained: bool = False
):
    if model_name == "AlexNet":
        return get_alexnet(
            num_classes = num_classes, 
            pretrained = pretrained
        )

    elif model_name == "ResNet50":
        return get_resnet50(
            num_classes = num_classes, 
            pretrained = pretrained
        )

    elif model_name == "ResNet18":
        return get_resnet18(
            num_classes = num_classes, 
            pretrained = pretrained
        )
    
    elif model_name == "ResNet101":
        return get_resnet101(
            num_classes = num_classes, 
            pretrained = pretrained
        )
    
    elif model_name == "NIN":
        return get_nin(
            num_classes = num_classes, 
            pretrained = pretrained
        )

    elif model_name == "VGG16":
        return get_vgg16(
            num_classes = num_classes, 
            pretrained = pretrained
        )
    
    elif model_name == "GoogleLenet":
        return get_googlenet(
            num_classes = num_classes, 
            pretrained = pretrained
        )