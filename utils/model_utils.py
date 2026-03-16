import os
import torch


def load_model_params(model, args: dict):
    """
    Load the parameters for the model.
    :param model:
    :param args: model configurations
    :return:
    """
    loaded_model_path = os.path.join(args['source']['directory'], args['source']['filename'])
    checkpoint = torch.load(loaded_model_path, map_location=torch.device('cpu'), weights_only=True)
    # checkpoint_paras = checkpoint["model_state_dict"]
    # filtered_state_dict = {k: v for k, v in checkpoint_paras.items() if
    #                        k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(checkpoint["model_state_dict"], strict=False) # strict=False is imperative if adapt the model training to different label levels
    print(f"Custom pretrained weights loaded from {loaded_model_path}")


def load_from_supcon_checkpoint(model, checkpoint):
    """
    Loads a standard timm Swin Transformer and its final classifier layer
    from a custom SupConSwinT checkpoint.

    Assumes:
      - checkpoint['model_state_dict'] has keys of the form 'encoder.*' for
        the Swin backbone, and 'head.*' for the projection head (which we skip).
      - checkpoint['classifier_state_dict'] has keys 'fc.weight' and 'fc.bias'
        for the final linear classifier.
      - We want to load everything into a timm Swin model that has
        'layers.*' for the backbone and 'head.weight'/'head.bias' for the final layer.

    :param model:The model for loading the weights.
    :param checkpoint: SupCon checkpoint.
    """

    # # 1. Load the SupConSwinT checkpoint
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 2. Convert the encoder weights -> standard Swin
    supcon_encoder_dict = checkpoint['model_state_dict']
    new_swin_dict = {}

    for key, value in supcon_encoder_dict.items():
        # Only use weights from 'encoder.*'
        if key.startswith('encoder.'):
            # Remove the 'encoder.' prefix
            stripped_key = key.replace('encoder.', '', 1)

            # Skip any 'head.*' (the projection head, not part of standard Swin)
            if stripped_key.startswith('head.'):
                continue

            # Put this into the new dict
            new_swin_dict[stripped_key] = value

    # 3. Convert the classifier weights -> Swin final classifier (model.head)
    supcon_classifier_dict = checkpoint['classifier_state_dict']
    # The original classifier keys are 'fc.weight' and 'fc.bias'
    # Timm SwinTransformer final classifier keys are 'head.weight' and 'head.bias'
    new_swin_dict['head.fc.weight'] = supcon_classifier_dict['fc.weight']
    new_swin_dict['head.fc.bias'] = supcon_classifier_dict['fc.bias']

    # 5. Load into the timm model
    missing, unexpected = model.load_state_dict(new_swin_dict, strict=False)

    print("Missing keys when model loading:", missing)
    print("Unexpected keys in state_dict:", unexpected)

    return model
