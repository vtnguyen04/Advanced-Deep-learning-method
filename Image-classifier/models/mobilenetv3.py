
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Callable

def _make_divisible(v: float, divisor: int, min_value = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace: bool = True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Callable[..., nn.Module] = None,
        activation_layer: Callable[..., nn.Module] = None,
        dilation: int = 1,
    ):
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, 
                      dilation = dilation, groups = groups, bias = False),
            norm_layer(out_planes),
            activation_layer(inplace = True)
        )
        self.out_channels = out_planes

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        use_se: bool,
        use_hs: bool,
        norm_layer: Callable[..., nn.Module] = None,
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.out_channels = oup
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNActivation(inp, hidden_dim, kernel_size = 1, norm_layer = norm_layer,
                                          activation_layer = h_swish if use_hs else nn.ReLU))
        
        # dw
        layers.append(ConvBNActivation(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, 
                                      groups = hidden_dim, norm_layer = norm_layer,
                                      activation_layer = h_swish if use_hs else nn.ReLU))
        
        if use_se:
            layers.append(SELayer(hidden_dim))

        # pw-linear
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias = False))
        layers.append(norm_layer(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(
        self,
        cfgs: List,
        mode: str,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        norm_layer: Callable[..., nn.Module] = None,
        dropout: float = 0.2,
        round_nearest: int = 8,
    ):
        super(MobileNetV3, self).__init__()
        # Set default norm layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # setting of inverted residual blocks
        # cfgs format: kernel, expansion, out_ch, use_se, use_hs, stride
        
        # building first layer
        input_channel = _make_divisible(16 * width_mult, round_nearest)
        layers = [ConvBNActivation(3, input_channel, 
                                   stride = 2, 
                                   norm_layer = norm_layer, 
                                   activation_layer = h_swish)]
        
        # building inverted residual blocks
        block = InvertedResidual
        for k, exp, c, use_se, use_hs, s in cfgs:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            layers.append(block(input_channel, output_channel, k, s, exp, use_se, use_hs, norm_layer))
            input_channel = output_channel
            
        # building last several layers
        if mode == 'large':
            last_conv = _make_divisible(960 * width_mult, round_nearest)
            layers.append(ConvBNActivation(input_channel, last_conv, 
                                           kernel_size = 1,  
                                           norm_layer = norm_layer, 
                                           activation_layer = h_swish))
            last_channel = 1280
        else:
            last_conv = _make_divisible(576 * width_mult, round_nearest)
            layers.append(ConvBNActivation(input_channel, last_conv, 
                                           kernel_size = 1, 
                                           norm_layer = norm_layer, 
                                           activation_layer = h_swish))
            last_channel = 1024
            
        self.features = nn.Sequential(*layers)
        
        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv, last_channel),
            h_swish(inplace = True),
            nn.Dropout(p = dropout),
            nn.Linear(last_channel, num_classes),
        )
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                
    def _forward_impl(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def forward(self, x):
        return self._forward_impl(x)

def mobilenet_v3_large(num_classes: int = 1000, width_mult: float = 1.0):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 0, 0, 1],
        [3, 4, 24, 0, 0, 2],
        [3, 3, 24, 0, 0, 1],
        [5, 3, 40, 1, 0, 2],
        [5, 3, 40, 1, 0, 1],
        [5, 3, 40, 1, 0, 1],
        [3, 6, 80, 0, 1, 2],
        [3, 2.5, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [5, 6, 160, 1, 1, 2],
        [5, 6, 160, 1, 1, 1],
        [5, 6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, 'large', num_classes, width_mult)

def mobilenet_v3_small(num_classes: int = 1000, width_mult: float = 1.0):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 2],
        [5, 6, 96, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],
    ]
    return MobileNetV3(cfgs, 'small', num_classes, width_mult)

def get_mobilenetv3(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    width_mult: float = 1.0
):
    if pretrained:
        if 'large' in model_name.lower():
            model = models.mobilenet_v3_large(pretrained = True)
        else:
            model = models.mobilenet_v3_small(pretrained = True)
        for param in model.parameters(): 
            param.requires_grad = False
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    else:
        if 'large' in model_name.lower():
            model = mobilenet_v3_large(num_classes, width_mult)
        else:
            model = mobilenet_v3_small(num_classes, width_mult)
    return model

def get_mobilenet_v3_large(
    num_classes: int,
    pretrained: bool = False,
    width_mult: float = 1.0
):
    return get_mobilenetv3(
        'mobilenet_v3_large',
        num_classes,
        pretrained,
        width_mult
    )

def get_mobilenet_v3_small(
    num_classes: int,
    pretrained: bool = False,
    width_mult: float = 1.0
):
    return get_mobilenetv3(
        'mobilenet_v3_small',
        num_classes,
        pretrained,
        width_mult
    )
