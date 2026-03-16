import os

import torch
from torch.optim.lr_scheduler import (
ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
)
from utils import REASSIGN_LABEL_NAME_L3, REASSIGN_NAME_LABEL_L3L2, NAME_LABEL_L2, NAME_ABB_L2, REASSIGN_NAME_LABEL_L3
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from sklearn.manifold import TSNE
import umap


scheduler_dict = {
    'cosine': CosineAnnealingLR,
    'reduced': ReduceLROnPlateau,
    # 'exp': ExponentialLR,
    # 'coswarm': CosineAnnealingWarmRestarts
}

def get_model_output(model, inputs):
    """
    This is the helper function to extract the output of the model, unify models from different sources:
    torchvision models and hugging face models
    :param model: the model
    :param inputs: data
    :return: logits
    """

    outputs = model(inputs)

    # Deal with hugging face model and torchvision model separately
    if hasattr(outputs, "logits"):
        return outputs.logits
    else:
        return outputs

class AverageMeter(object):
    """
    Computes and stores the average and current value,
    copied from https://github.com/HobbitLong/SupContrast/blob/master/util.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def l3_to_l2(batch_l3_pred, batch_metadata, device):
    """
    Calculate the L2 correct predictions in a batch data. Converting from L3 predictions.
    :param batch_l3_pred: L3 predicted labels
    :param batch_metadata: Batch metadata
    :param device:
    :return:
    """
    batch_labels_l2 = batch_metadata['l2_label'].to(device)
    batch_predicted_l2 = []
    for l3_label in batch_l3_pred:
        l3_word_label = REASSIGN_LABEL_NAME_L3[l3_label.item()]
        l2_label = REASSIGN_NAME_LABEL_L3L2[l3_word_label][1]
        batch_predicted_l2.append(l2_label)
    batch_predicted_l2 = torch.tensor(batch_predicted_l2, device=batch_l3_pred.device)
    correct_l2 = batch_predicted_l2.eq(batch_labels_l2).sum()
    return correct_l2

def feat_extraction(model, dl, desc, device):
    """
    Extract features from model.
    :param model: Model to extract features for targeted data, should be set to eval mode before passing.
    :param dl: Dataloader for extracting the features.
    :param desc: Description for the tqdm progress bar.
    :param device:
    :return: features from the encoder, labels, file names.
    """
    def _feat_extraction(feat_list, label_list, filename_list):
        for images, targets, metadata in tqdm(dl, desc=desc):
            images = images.to(device)

            # metadata['file_name'] = ["image1.jpg", "image2.jpg", ...]
            file_names = metadata['file_name']

            outputs = model(images)  # Model has been adapted in the load_model function for feature extraction
            feat_list.append(outputs.detach().cpu().numpy())
            label_list.extend(targets.numpy())
            filename_list.extend(file_names)
        return feat_list, label_list, filename_list

    labels = []
    features = []
    all_filenames = []
    with torch.no_grad():
        features, labels, all_filenames = _feat_extraction(
            features, labels, all_filenames
        )

    features_np = np.concatenate(features, axis=0)
    labels = np.array(labels)

    return features_np, labels, all_filenames

def draw_latent(feats, labels, fig_name: str, use_l2: bool = False, main_hab: bool = True):
    """
    Draw the t-sne figures for 2-D features
    :param feats: 2-D features from t-sne
    :param labels: Labels for the feats.
    :param use_l2: For model trained on L2 labels, only l2 habitats are displayed.
    :param fig_name: The name of the latent space, t-sne of umap
    :param main_hab: if only major l2 habitats are recorded.
    :return:
    """
    def draw_figs(feat_2d, fig_size: tuple, select_cls: list, hab_name: str):
        """
        Draw and save the latent space figure.
        :param feat_2d: the 2D features from tsne or umap, numpy array with shape of (num_samples, 2)
        :param fig_size: Individual figure size, tuple ()
        :param select_cls: Which classes are selected for showing on the t-sne figure
        :param hab_name: Use l2 habitat labels or "ALL".
        :return:
        """
        plt.figure(figsize=fig_size)
        unique_labels = np.unique(labels)

        # Use a colormap that has at least 20 distinct colors (e.g., tab20)
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))

        # Define the conversion dict to convert the numerical label to word label
        LABEL_NAME_L2 = {value: key for key, value in NAME_LABEL_L2.items()}
        conversion_dict = LABEL_NAME_L2 if use_l2 else REASSIGN_LABEL_NAME_L3

        for i, class_id in enumerate(unique_labels):
            word_label = conversion_dict[class_id]
            if word_label in select_cls:
                idx = (labels == class_id)
                color = cmap(i)
                plt.scatter(feat_2d[idx, 0], feat_2d[idx, 1], label=word_label, s=5, color=color)

        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        # plt.legend()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        wandb.log({f"{hab_name}-{fig_name}": wandb.Image(plt)})
        plt.close()

    # Save all classes
    word_label_all = list(NAME_LABEL_L2.keys()) if use_l2 else list(REASSIGN_NAME_LABEL_L3.keys())
    draw_figs(feats, fig_size=(10, 8), select_cls=word_label_all, hab_name='ALL')

    # Save L3 classes with the save L2 label, only if training with L3 label
    if not use_l2:
        main_hab_list = ['GL', 'WLF', 'WL', 'HS']
        for l2_word_label, l2_label in NAME_LABEL_L2.items():
            l3_word_label_list = [k for k, v in REASSIGN_NAME_LABEL_L3L2.items() if v[1]==l2_label]
            abb_l2_name = NAME_ABB_L2[l2_word_label]

            # Draw only if `main_hab` is False OR if `abb_l2_name` is in the main list
            if not main_hab or abb_l2_name in main_hab_list:
                draw_figs(feats, fig_size=(10, 8), select_cls=l3_word_label_list, hab_name=abb_l2_name)

def save_emb(feats, labels, filenames, method, save_dir):
    df = pd.DataFrame({
        f'{method}_x': feats[:, 0],
        f'{method}_y': feats[:, 1],
        'label': labels,
        'filename': filenames
    })
    save_path = os.path.join(save_dir, f"{method}_results.csv")
    df.to_csv(save_path, index=False)

def feat_reduction(feats, labels, method: str = 'tsne'):
    """
    Reduce the dimension of the feats
    :param feats: encoder produced embeddings
    :param labels: ground truth labels, used by umap
    :param method: umap or tsne
    :return: the reduced feats
    """
    if method == 'tsne':
        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30,
            max_iter=1000)
    elif method == 'umap':
        reducer = umap.UMAP(
            n_neighbors=15,  # Balances local/global structure
            min_dist=0.1,  # Controls how close points can get
            n_components=2,  # Project down to 2D
            metric='euclidean',  # Distance metric
            random_state=42
        )
    else:
         raise ValueError(f'{method} is not supported for feature dimension reduction.')

    reduced_feats = reducer.fit_transform(feats)
    return reduced_feats

class UmapLearning(object):
    """
    Reducing the dimension of the input feats using t-sne or umap.
    """
    def __init__(self, train_data, train_labels, test_data):
        """
        :param train_data: np array containing original embeddings on the training data.
        :param train_labels: np array containing labels for the training data.
        :param test_data: np array containing original embeddings on the test/valid data.
        """
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.mapper = self._train()

    def _train(self):
        """
        Learn on training data
        :return:
        """
        mapper = umap.UMAP(
            n_neighbors=15,  # Balances local/global structure
            min_dist=0.1,  # Controls how close points can get
            n_components=2,  # Project down to 2D
            metric='euclidean',  # Distance metric
            random_state=42
        ).fit(self.train_data)
        return mapper

    def _test(self):
        return self.mapper.transform(self.test_data)

    def get_emb(self):
        """
        Get the reduced embeddings after UMAP.
        :return:
        """
        train_emb = self.mapper.embedding_
        test_emb = self._test()
        return train_emb, test_emb
