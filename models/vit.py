from transformers import SwinForImageClassification, AutoModelForImageClassification
from utils import REASSIGN_LABEL_NAME_L3, REASSIGN_NAME_LABEL_L3
import timm
import logging


# class MyAutoSwinT(AutoModelForImageClassification):
#     def forward(self, *args, **kwargs):
#         outputs =super().forward(*args, **kwargs)
#         return outputs.logits

def swint_hugg(args: dict):
    """
    This function can fail if loaded when training on L2 labels, it needs to re-adapt
    :param args:
    :return:
    """
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        label2id=REASSIGN_NAME_LABEL_L3,
        id2label=REASSIGN_LABEL_NAME_L3,
        ignore_mismatched_sizes=True,
    )
    return model

def swint(args: dict):
    """
    Load the swint model created by the timm package
    :param args: The configurations for the model
    :return: The created swint model
    """
    model_names = {
        224: {
            "tiny": "microsoft/swin_tiny_patch4_window7_224",
            "small": "microsoft/swin_small_patch4_window7_224",
            "base": "microsoft/swin_base_patch4_window7_224",
            "large": "microsoft/swin_large_patch4_window7_224"
        },
        384: {
            "base": "microsoft/swin_base_patch4_window12_384",
            "large": "microsoft/swin_large_patch4_window12_384"
        }
    }

    # Checking valid input sizes
    if args['input_size'] in model_names:
        input_size, model_size = args['input_size'], args['model_config']
        model_name = model_names[input_size][model_size]
    else:
        supported_sizes = ", ".join(map(str, model_names.keys()))
        raise KeyError(f"Invalid model input size. Supported sizes are: {supported_sizes}.")

    model = timm.create_model(model_name, pretrained=args['pretrained'], num_classes=args['num_classes'])

    # If only fine tune the head of the swint
    if args.get('fix_body', False):
        for name, param in model.named_parameters():
            if name.startswith("head"):  # Typically, the classifier is named 'head' in timm models
                param.requires_grad = True
            else:
                param.requires_grad = False

        print('The body parameters of the loaded model has been frozen for training!')
    return model

def vit(args: dict):
    """
    Load the vit model created by the timm package
    :param args: The configurations for the model
    :return: The created swint model
    """
    model_names = {
        224: {
            "base": "vit_base_patch16_siglip_224",
        },
        384: {
            "large": "vit_large_patch16_siglip_384",
        }
    }

    # Checking valid input sizes
    if args['input_size'] in model_names:
        input_size, model_size = args['input_size'], args['model_config']
        model_name = model_names[input_size][model_size]
    else:
        supported_sizes = ", ".join(map(str, model_names.keys()))
        raise KeyError(f"Invalid model input size. Supported sizes are: {supported_sizes}.")

    model = timm.create_model(model_name, pretrained=args['pretrained'], num_classes=args['num_classes'])

    # If only fine tune the head of the swint
    if args.get('fix_body', False):
        for name, param in model.named_parameters():
            if name.startswith("head"):  # Typically, the classifier is named 'head' in timm models
                param.requires_grad = True
            else:
                param.requires_grad = False

        print('The body parameters of the loaded model has been frozen for training!')
    return model

def resnet(args: dict):
    """
    ResNet is put inside the ViT lib because it is loaded from Hugging Face Hub.
    Load the resent model created by the timm package
    :param args: The configurations for the model
    :return: The created resnet model
    """
    model_lib = {
        "50_1": "resnetv2_50x1_bit.goog_in21k_ft_in1k",
        "101_1": "resnetv2_101x1_bit.goog_in21k_ft_in1k",
        "152_2": "resnetv2_152x2_bit.goog_in21k_ft_in1k"
    }

    # Checking valid input sizes
    model_config = args.get('model_config', '50_1')

    if model_config in model_lib:
        model_name = model_lib[model_config]
    else:
        supported_config = ", ".join(map(str, model_lib.keys()))
        raise KeyError(f"Invalid resnet config. Supported size configs are: {supported_config}.")

    model = timm.create_model(model_name, pretrained=args['pretrained'], num_classes=args['num_classes'])

    # If only fine tune the head of the swint
    if args.get('fix_body', False):
        for name, param in model.named_parameters():
            if name.startswith("head"):  # Typically, the classifier is named 'head' in timm models
                param.requires_grad = True
            else:
                param.requires_grad = False

        print('The body parameters of the loaded model has been frozen for training!')
    return model