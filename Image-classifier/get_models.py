from models.AlexNet import get_alexnet
from models.NIN import get_nin
from models.mobilenet import get_mobilenet_v1
from models.mobileNetV2 import get_mobilenet_v2
from models.mobilenetv3 import get_mobilenet_v3_large, get_mobilenet_v3_small
from models.densenet import get_densenet121, get_densenet169, get_densenet201
from models.resnet import get_resnet50, get_resnet101, get_resnet18
from models.vgg16 import get_vgg16
from models.googlelenet import get_googlenet
from models.efficientNet import get_efficientnet_b0, get_efficientnet_b1, get_efficientnet_b2, get_efficientnet_b3
from models.NasNet import get_nasnet_small, get_nasnet_large
from models.SENet import get_se_resnet101, get_se_resnet152, get_se_resnet50
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
    
    elif model_name == "MobileNetv1":
        return get_mobilenet_v1(
            num_classes = num_classes, 
            pretrained = pretrained
        )
    
    elif model_name == "MobileNetv2":
        return get_mobilenet_v2(
            num_classes = num_classes, 
            pretrained = pretrained
        )
    elif model_name == "MobileNetv3-large":
        return get_mobilenet_v3_large(
            num_classes = num_classes, 
            pretrained = pretrained
        )

    elif model_name == "MobileNetv3-small":
        return get_mobilenet_v3_small(
            num_classes = num_classes, 
            pretrained = pretrained
        )
    
    elif model_name == "DenseNet169":
        return get_densenet169(
            num_classes = num_classes, 
            pretrained = pretrained
        )
    elif model_name == "DenseNet121":
        return get_densenet121(
            num_classes = num_classes, 
            pretrained = pretrained
        )
    elif model_name == "DenseNet201":
        return get_densenet201(
            num_classes = num_classes, 
            pretrained = pretrained
        )

    elif model_name == "EfficientNet-b0":
        return get_efficientnet_b0(
            num_classes = num_classes, 
            pretrained = pretrained
        )

    elif model_name == "EfficientNet-b1":
        return get_efficientnet_b1(
            num_classes = num_classes, 
            pretrained = pretrained
        )

    elif model_name == "EfficientNet-b2":
        return get_efficientnet_b2(
            num_classes = num_classes, 
            pretrained = pretrained
        )
    
    elif model_name == "EfficientNet-b3":
        return get_efficientnet_b3(
            num_classes = num_classes, 
            pretrained = pretrained
        )
    
    elif model_name == "NasNet-mobile":
        return get_nasnet_small(
            num_classes = num_classes, 
            pretrained = pretrained
        )

    elif model_name == "NasNet-large":
        return get_nasnet_large(
            num_classes = num_classes, 
            pretrained = pretrained
        )

    elif model_name == "SeResNet50":
        return get_se_resnet50(
            num_classes = num_classes, 
            pretrained = pretrained
        )

    elif model_name == "SeResNet101":
        return get_se_resnet101(
            num_classes = num_classes, 
            pretrained = pretrained
        )

    elif model_name == "SeResNet152":
        return get_se_resnet152(
            num_classes = num_classes, 
            pretrained = pretrained
        )
