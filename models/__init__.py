# models/__init__.py

from .cnns import wrn, effn_v2, resnext, WRNCustom, DVResNext
from .vit import swint, resnet, vit
from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

model_latent_dim = {
    'swint': {'tiny': 768, 'small': 768, 'base': 1024, 'large': 1536},
    'wrn': {'50_2': 2048, '101_2': 2048},
    'resnext': {'50': 2048, '101': 2048},
    'efficientnet': {'small': 1280, 'medium': 1280, 'large': 1280},
}


def create_model(model_name: str, cv_id: int, args: dict, verbose=True):
    """
    Creates and returns the model based on the model name.
    :param model_name: The name of the model. remove this argument later, we do not need it.
    :param cv_id: The cross validation id.
    :param args: Configurations for the model.
    :param verbose: If show model info.
    :return:
    """
    if model_name == 'wrn':
        model = wrn(args)
    elif model_name == 'efficientnet':
        model = effn_v2(args)
    elif model_name == 'resnext':
        model = resnext(args)
    elif model_name == 'dv_resnext':
        model = DVResNext(args)
    elif model_name == 'swint':
        model = swint(args)
    elif model_name == 'resnet':
        model = resnet(args)
    elif model_name == 'vit':
        model = vit(args)
    else:
        raise ValueError(f'Model {model_name} not defined.')

    # Print the model details
    if cv_id == 0 and verbose:
        summary(model, input_size=(8, 3, args['input_size'], args['input_size']))

    return model

class SupConEncoder(nn.Module):
    """
    Generic encoder + projection head for SupCon.
    Works with timm Swin/ResNet variants and torchvision CNN backbones
    (WRN, ResNeXt, EfficientNetV2) by stripping the classifier layer.
    """
    def __init__(self, model_name: str, cv_id: int, args: dict, head: str = 'mlp', proj_dim: int = 128):
        super().__init__()
        backbone = create_model(model_name, cv_id, args, verbose=False)

        # Strip classifier for feature extraction
        if hasattr(backbone, 'reset_classifier'):
            backbone.reset_classifier(0)
        else:
            if hasattr(backbone, 'fc') and isinstance(backbone.fc, nn.Module):
                backbone.fc = nn.Identity()
            elif hasattr(backbone, 'classifier') and isinstance(backbone.classifier, nn.Module):
                backbone.classifier = nn.Identity()
            elif hasattr(backbone, 'model') and hasattr(backbone.model, 'fc') and isinstance(backbone.model.fc, nn.Module):
                # DVResNext wrapper compatibility
                backbone.model.fc = nn.Identity()
            else:
                raise ValueError('Unsupported backbone: cannot strip classifier for SupCon.')

        self.encoder = backbone

        # Resolve feature dimension
        dim_in = None
        name = args['name']
        config = args.get('model_config')
        if name in model_latent_dim and config in model_latent_dim[name]:
            dim_in = model_latent_dim[name][config]
        if dim_in is None:
            # Fallback: infer dynamically with a dummy forward
            input_size = args.get('input_size', 224)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, input_size, input_size)
                feats = self.encoder(dummy)
                if isinstance(feats, dict) and 'logits' in feats:
                    feats = feats['logits']
                dim_in = feats.shape[1]

        if head == 'linear':
            self.head = nn.Linear(dim_in, proj_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, proj_dim)
            )
        else:
            raise NotImplementedError(f'head not supported: {head}')

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class SupConSwinT(nn.Module):
    """
    SwinT encoder + projection head, copied from the https://github.com/HobbitLong/SupContrast/blob/master/networks/resnet_big.py
    """
    def __init__(self, model_name, cv_id, args: dict, head='mlp', feat_dim=128):
        """
        Initialise the SwinT model for the SupCon method
        :param model_name: model name for create the model, only accept SwinT
        :param cv_id: set to 0, only used for torchinfo summary in the create_model function
        :param args: model args from the main args
        :param head: the configuration for the projection head on top of the backbone
        :param feat_dim: the output dimension of the head
        """
        super(SupConSwinT, self).__init__()

        # Set up SwinT backbone
        backbone = create_model(model_name, cv_id, args, verbose=False)
        if isinstance(backbone, timm.models.swin_transformer.SwinTransformer):
            # remove the linear classifier from the SwinT backbone
            backbone.reset_classifier(0)
        else:
            raise ValueError("Unsupported model type for SupCon. Only SwinT is supported.")
        self.encoder = backbone

        # Set up the projection header
        model_name, model_config = args['name'], args['model_config']
        dim_in = model_latent_dim[model_name][model_config]

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class LinearClassifier(nn.Module):
    """
    Linear classifier for the SupCon,
    copied from the https://github.com/HobbitLong/SupContrast/blob/master/networks/resnet_big.py
    """
    def __init__(self, args: dict):
        super(LinearClassifier, self).__init__()

        model_name, model_config = args['name'], args['model_config']
        feat_dim = model_latent_dim[model_name][model_config]
        self.fc = nn.Linear(feat_dim, args['num_classes'])

    def forward(self, features):
        return self.fc(features)
