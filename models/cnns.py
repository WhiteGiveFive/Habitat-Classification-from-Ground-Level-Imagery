from torchvision.models import (
    wide_resnet50_2, Wide_ResNet50_2_Weights,
    wide_resnet101_2, Wide_ResNet101_2_Weights,
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
    resnext101_64x4d, ResNeXt101_64X4D_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
)
import torch.nn as nn
import torch


def wrn(args):
    """
    Wide ResNet50
    :param args:
    :return: adapted wide resnet
    """
    pretrained = args['pretrained']
    model_lib = {
        "50_2": (wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V1),
        "101_2": (wide_resnet101_2, Wide_ResNet101_2_Weights.IMAGENET1K_V1),
    }
    model_config = args.get('model_config', '50_2')
    if model_config in model_lib:
        wide_resnet = model_lib[model_config][0]
        pretrained_weights = model_lib[model_config][1]
    else:
        supported_config = ", ".join(map(str, model_lib.keys()))
        raise KeyError(f"Invalid wide resnet config. Supported size configs are: {supported_config}.")

    if pretrained:
        model = wide_resnet(weights=pretrained_weights)
    else:
        model = wide_resnet()

    model.fc = nn.Linear(in_features=2048, out_features=args['num_classes'])

    return model

def effn_v2(args):
    """
    EfficientNet v2
    :param args:
    :return: adapted efficientnet v2
    """
    pretrained = args['pretrained']
    model_lib = {
        "small": (efficientnet_v2_s, EfficientNet_V2_S_Weights.IMAGENET1K_V1),
        "medium": (efficientnet_v2_m, EfficientNet_V2_M_Weights.IMAGENET1K_V1),
        "large": (efficientnet_v2_l, EfficientNet_V2_L_Weights.IMAGENET1K_V1),
    }
    model_config = args.get('model_config', 'medium')
    if model_config in model_lib:
        efficientnet_v2 = model_lib[model_config][0]
        pretrained_weights = model_lib[model_config][1]
    else:
        supported_config = ", ".join(map(str, model_lib.keys()))
        raise KeyError(f"Invalid wide resnet config. Supported size configs are: {supported_config}.")

    if pretrained:
        model = efficientnet_v2(weights=pretrained_weights)
    else:
        model = efficientnet_v2()

    model.classifier[1] = nn.Linear(in_features=1280, out_features=args['num_classes'])

    return model

def resnext(args):
    """
    Torchvision ResNeXt family.
    :param args: expects keys 'model_config' in {'50','101'} and 'pretrained'.
    :return: adapted resnext model
    """
    model_lib = {
        "50": (resnext50_32x4d, ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
        "101": (resnext101_64x4d, ResNeXt101_64X4D_Weights.IMAGENET1K_V1),
    }
    model_config = args.get('model_config', '50')
    if model_config in model_lib:
        builder, weights_enum = model_lib[model_config]
    else:
        supported = ", ".join(model_lib.keys())
        raise KeyError(f"Invalid resnext config. Supported size configs are: {supported}.")

    pretrained = args.get('pretrained', False)
    model = builder(weights=weights_enum if pretrained else None)

    # Replace the classifier to match num_classes for standard training flows.
    # For SupCon, the caller may replace this with nn.Identity().
    model.fc = nn.Linear(model.fc.in_features, out_features=args['num_classes'])
    return model

def resnet101(args, if_pretrained=True):
    pass


class WRNCustom(nn.Module):
    def __init__(self, args: dict):
        super(WRNCustom, self).__init__()
        pretrained = args.get('pretrained', False)
        if pretrained:
            self.wrn50_2 = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            self.wrn50_2 = wide_resnet50_2()

        # Replace the last fully connected layer to match the number of classes
        self.wrn50_2.fc = nn.Linear(self.wrn50_2.fc.in_features, args['num_classes'])

    def forward(self, x):
        return self.wrn50_2(x)

class DVResNext(nn.Module):
    """
    Load the trained model from Andy's project--DeepVerge (DV).
    """
    def __init__(self, args: dict):
        super(DVResNext, self).__init__()
        model_lib = {
            "50": (resnext50_32x4d, ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
            "101": (resnext101_64x4d, ResNeXt101_64X4D_Weights.IMAGENET1K_V1),
        }
        model_config = args.get('model_config', '50')
        if model_config in model_lib:
            resnext = model_lib[model_config][0]
            pretrained_weights = model_lib[model_config][1]
        else:
            supported_config = ", ".join(map(str, model_lib.keys()))
            raise KeyError(f"Invalid resnext config. Supported size configs are: {supported_config}.")

        pretrained = args.get('pretrained', False)
        if pretrained:
            self.model = resnext(weights=pretrained_weights)
        else:
            self.model = resnext()

        self.model.fc = nn.Linear(self.model.fc.in_features, out_features=args['num_classes'])

        # self.resnext = resnext50_32x4d()
        # if pretrained:
        #     pretrained_path = './models/pretrained_models/670-dataset4-ResNeXt50.pth'
        #     self.resnext.fc = nn.Linear(self.resnext.fc.in_features, out_features=4)
        #     self.resnext.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location=torch.device('cpu')))
        #
        # self.resnext.fc = nn.Linear(self.resnext.fc.in_features, args['num_classes'])

    def forward(self, x):
        # return self.resnext(x)
        return self.model(x)
