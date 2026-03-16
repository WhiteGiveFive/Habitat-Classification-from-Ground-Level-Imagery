import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torcheval.metrics import MulticlassF1Score, MulticlassConfusionMatrix
import matplotlib
matplotlib.use('Agg')   # Set matplotlib to non-interactive mode, avoiding warning in sweep runs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging
import wandb

from utils import REASSIGN_LABEL_NAME_L3, REASSIGN_NAME_LABEL_L3L2
from utils.train_utils import get_model_output


def test_obliterated(testloader, model, num_cls: int):
    """
    Model evaluation function.
    :param testloader: validation or test dataloader
    :param model: model that has been offloaded to cuda
    :param num_cls: number of classes
    :return: a dict consists of evaluation metrics
    """
    model.eval()

    metrics = {
        'test_correct': 0,
        'test_loss': 0,
        'test_total': 0,
        'test_acc': 0,
        'f1_score': 0,
        'confusion_matrix': np.zeros((num_cls, num_cls)),
    }

    criterion = nn.CrossEntropyLoss().cuda()
    f1_metric = MulticlassF1Score(num_classes=num_cls, average='macro')
    cm_metric = MulticlassConfusionMatrix(num_classes=num_cls)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # accuracy
            _, predicted = torch.max(outputs, -1)
            correct = predicted.eq(labels).sum()

            metrics['test_correct'] += correct.item()
            metrics['test_loss'] += loss.item() * images.size(0)
            metrics['test_total'] += labels.size(0)

            # update F1 score and confusion matrix metrics
            f1_metric.update(predicted, labels)
            cm_metric.update(predicted, labels)

    metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']
    metrics['f1_score'] = f1_metric.compute().item()
    metrics['confusion_matrix'] = cm_metric.compute().numpy()
    return metrics


class Evaluator:
    def __init__(self, model, test_loader, num_classes, result_save_dir, cv_id, args):
        """
        Initialise the evaluator.
        :param model: the learned model
        :param test_loader: test dataloader
        :param num_classes: the number of classes
        """
        self.model = model
        self.test_loader = test_loader
        self.num_classes_l3, self.num_classes_l2 = num_classes
        self.result_save_dir = result_save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.cv_id = cv_id

        self.f1_metric_l3 = MulticlassF1Score(num_classes=self.num_classes_l3, average='weighted')
        self.f1_metric_l2 = MulticlassF1Score(num_classes=self.num_classes_l2, average='weighted')
        self.cm_metric_l3 = MulticlassConfusionMatrix(num_classes=self.num_classes_l3)
        self.cm_metric_l2 = MulticlassConfusionMatrix(num_classes=self.num_classes_l2)
        self.model.to(self.device)

        self.metrics = {
            'metrics_l3': {
                'test_correct': 0,
                'test_loss': 0.0,
                'test_total': 0,
                'top1_acc': 0,
                'top3_correct': 0,
                'top3_acc': 0,
                'f1_score': 0,
                'confusion_matrix': np.zeros((self.num_classes_l3, self.num_classes_l3)),
            },
            'metrics_l2': {
                'test_correct': 0,
                'test_total': 0,
                'top1_acc': 0,
                'f1_score': 0,
                'confusion_matrix': np.zeros((self.num_classes_l2, self.num_classes_l2)),
            }
        }
        self.misclassified = []
        self.accurate_classified = []

    def test(self):
        """
        Evaluate the model on the test set and compute the metrics
        :return: metric dict including top1, top3, F1, and confusion matrix
        """
        model = self.model
        model.eval()

        metrics = self.metrics

        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for batch_idx, (images, labels, metadata) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = get_model_output(model, images)
                loss = criterion(outputs, labels)

                # Top-1 accuracy
                _, predicted = torch.max(outputs, 1)
                correct = predicted.eq(labels).sum()
                metrics['metrics_l3']['test_correct'] += correct.item()

                # Top-3 metrics
                top3_correct, top3_pred_indices, top3_probs= self._top3_metrics(outputs, labels)

                # Track the misclassified samples
                self._track_classification(predicted, labels, top3_pred_indices, top3_probs, metadata)

                metrics['metrics_l3']['top3_correct'] += top3_correct.item()

                # Update loss and number of samples
                metrics['metrics_l3']['test_loss'] += loss.item() * images.size(0)
                metrics['metrics_l3']['test_total'] += labels.size(0)

                # Update F1 score and confusion matrix
                self.f1_metric_l3.update(predicted, labels)
                self.cm_metric_l3.update(predicted, labels)

                # If record performance on L2 labels
                if self.args.get('l2_metrics', False):
                    self._update_l2_metrics(batch_output=outputs, batch_metadata=metadata)

        # Calculate the overall accuracy and metrics
        metrics['metrics_l3']['top1_acc'] = metrics['metrics_l3']['test_correct'] / metrics['metrics_l3']['test_total']
        metrics['metrics_l3']['top3_acc'] = metrics['metrics_l3']['top3_correct'] / metrics['metrics_l3']['test_total']
        metrics['metrics_l3']['f1_score'] = self.f1_metric_l3.compute().item()
        metrics['metrics_l3']['confusion_matrix'] = self.cm_metric_l3.compute().cpu().numpy()

        if self.args.get('l2_metrics', False):
            metrics['metrics_l2']['top1_acc'] = metrics['metrics_l2']['test_correct'] / metrics['metrics_l2']['test_total']
            metrics['metrics_l2']['f1_score'] = self.f1_metric_l2.compute().item()
            metrics['metrics_l2']['confusion_matrix'] = self.cm_metric_l2.compute().cpu().numpy()

        return metrics

    def test_classifier(self, classifier):
        """
        Evaluate the classifier trained after SupCon pretraining on the test set and compute the metrics.
        :param classifier: The classifier to test.
        :return: Metric dict including top1, top3, F1, and confusion matrix.
        """
        model = self.model
        classifier.eval()
        model.eval()

        metrics = self.metrics

        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for batch_idx, (images, labels, metadata) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = get_model_output(classifier, model.encoder(images))
                loss = criterion(outputs, labels)

                # Top-1 accuracy
                _, predicted = torch.max(outputs, 1)
                correct = predicted.eq(labels).sum()
                metrics['metrics_l3']['test_correct'] += correct.item()

                # Top-3 metrics
                top3_correct, top3_pred_indices, top3_probs= self._top3_metrics(outputs, labels)

                # Track the misclassified samples
                self._track_classification(predicted, labels, top3_pred_indices, top3_probs, metadata)

                metrics['metrics_l3']['top3_correct'] += top3_correct.item()

                # Update loss and number of samples
                metrics['metrics_l3']['test_loss'] += loss.item() * images.size(0)
                metrics['metrics_l3']['test_total'] += labels.size(0)

                # Update F1 score and confusion matrix
                self.f1_metric_l3.update(predicted, labels)
                self.cm_metric_l3.update(predicted, labels)

                # If record performance on L2 labels
                if self.args.get('l2_metrics', False):
                    self._update_l2_metrics(batch_output=outputs, batch_metadata=metadata)

        # Calculate the overall accuracy and metrics
        metrics['metrics_l3']['top1_acc'] = metrics['metrics_l3']['test_correct'] / metrics['metrics_l3']['test_total']
        metrics['metrics_l3']['top3_acc'] = metrics['metrics_l3']['top3_correct'] / metrics['metrics_l3']['test_total']
        metrics['metrics_l3']['f1_score'] = self.f1_metric_l3.compute().item()
        metrics['metrics_l3']['confusion_matrix'] = self.cm_metric_l3.compute().cpu().numpy()

        if self.args.get('l2_metrics', False):
            metrics['metrics_l2']['top1_acc'] = metrics['metrics_l2']['test_correct'] / metrics['metrics_l2']['test_total']
            metrics['metrics_l2']['f1_score'] = self.f1_metric_l2.compute().item()
            metrics['metrics_l2']['confusion_matrix'] = self.cm_metric_l2.compute().cpu().numpy()

        return metrics

    def _update_l2_metrics(self, batch_output, batch_metadata):
        """
        Test the model performance on L2 labels, the model is still trained on L3 labels
        :param batch_output:
        :param batch_metadata:
        :return:
        """
        metrics = self.metrics
        batch_labels_l2 = batch_metadata['l2_label'].to(self.device)    # Ground truth labels
        _, batch_predicted_l3 = torch.max(batch_output, 1)

        # Convert the batch_predicted labels from L3 to L2: num_L3 -> word_L3 -> num_L2
        # batch_predicted_l3 = batch_predicted_l3.cpu().numpy()
        batch_predicted_l2 = []
        for l3_label in batch_predicted_l3:
            l3_word_label = REASSIGN_LABEL_NAME_L3[l3_label.item()]
            l2_label = REASSIGN_NAME_LABEL_L3L2[l3_word_label][1]
            batch_predicted_l2.append(l2_label)
        batch_predicted_l2 = torch.tensor(batch_predicted_l2, device=batch_predicted_l3.device)

        correct = batch_predicted_l2.eq(batch_labels_l2).sum()
        metrics['metrics_l2']['test_correct'] += correct.item()
        metrics['metrics_l2']['test_total'] += batch_labels_l2.size(0)

        # Update F1 score and confusion matrix
        self.f1_metric_l2.update(batch_predicted_l2, batch_labels_l2)
        self.cm_metric_l2.update(batch_predicted_l2, batch_labels_l2)

    def _top3_metrics(self, outputs, labels):
        top3_pred_indices = torch.topk(outputs, 3, dim=1).indices

        # Convert logits to probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Get the probabilities corresponding to the top-3 indices
        top3_probs = torch.gather(probabilities, 1, top3_pred_indices)

        # top3_pred_indices = top3_pred_indices.cpu().numpy()
        # top3_probs = top3_probs.cpu().numpy()
        top3_correct = torch.sum(torch.any(top3_pred_indices == labels.unsqueeze(1), dim=1))
        return top3_correct, top3_pred_indices, top3_probs

    def _track_classification(self, predictions, labels, top3_labels, top3_probs, metadata):
        """
        :param predictions: predicted labels in a batch data
        :param labels: the ground truth labels in a batch data
        :param top3_labels: top 3 labels in a batch data
        :param top3_probs: top 3 probabilities in a batch data
        :param metadata: the metadata associated to the batch data
        :return: a list of misclassified sample details
        """
        for i in range(len(labels)):
            instance_result = {
                'file_name': metadata['file_name'][i],
                'ground_truth_num_label': labels[i].item(),
                'ground_truth_word_label': metadata['plot_word_label'][i],
                'predicted_label': predictions[i].item(),
                'predicted_word_label': REASSIGN_LABEL_NAME_L3[predictions[i].item()],
                'top3_predictions': [
                    {'label': int(top3_labels[i][j]), 'probability': float(top3_probs[i][j])}
                    for j in range(3)
                ],
                'dataset': metadata['image_source'][i]
            }
            # # Add top-3 predictions and probabilities
            # instance_result['top3_predictions'] = [
            #     {'label': int(top3_labels[i][j]), 'probability': float(top3_probs[i][j])}
            #     for j in range(3)
            # ]
            if predictions[i] != labels[i]:
                self.misclassified.append(instance_result)
            else:
                self.accurate_classified.append(instance_result)

    def save_misclassified(self):
        """
        Save the misclassified samples to a CSV file.
        :return:
        """
        cv_id = self.cv_id

        def flatten_instance_result(instance_results, save_file_path):
            """
            Flatten the nested top-3 predictions into a format suitable for CSV.
            :param instance_results: List of instance results (either misclassified or accurate).
            :param save_file_path: Saved path for the output CSV file.
            """
            flat_data = []
            for instance in instance_results:
                # Base information
                base_info = {
                    'file_name': instance['file_name'],
                    'ground_truth_num_label': instance['ground_truth_num_label'],
                    'ground_truth_word_label': instance['ground_truth_word_label'],
                    'predicted_label': instance['predicted_label'],
                    'predicted_word_label': instance['predicted_word_label'],
                    'dataset': instance['dataset'],
                }

                # Flatten top-3 predictions
                for i, top3_entry in enumerate(instance['top3_predictions']):
                    base_info[f'top3_label_{i + 1}'] = top3_entry['label']
                    base_info[f'top3_prob_{i + 1}'] = top3_entry['probability']

                flat_data.append(base_info)

            # Convert to DataFrame and save
            df = pd.DataFrame(flat_data)
            df.to_csv(save_file_path, index=False)
            return df

        if self.misclassified:
            mis_file_path = os.path.join(self.result_save_dir, f'misclassified_samples_cv{cv_id}.csv')
            mis_df = flatten_instance_result(self.misclassified, mis_file_path)
            logging.info(f'Misclassified samples saved to {mis_file_path}')
            wandb.log({"Misclassifications": wandb.Table(dataframe=mis_df)})
        else:
            logging.info('No misclassified samples')

        if self.accurate_classified:
            cor_file_path = os.path.join(self.result_save_dir, f'correctly_classified_samples_cv{cv_id}.csv')
            cor_df = flatten_instance_result(self.accurate_classified, cor_file_path)
            logging.info(f'Correctly classified samples saved to {cor_file_path}')
            wandb.log({"Corclassifications": wandb.Table(dataframe=cor_df)})
        else:
            logging.info('No correctly classified samples')

    def save_cm(self, class_names_l3: list, class_names_l2: list) -> None:
        """
        Plots the con confusion matrix using Seaborn and Matplotlib
        :param class_names_l3: List of class names for level 3
        :param class_names_l2: List of class names for level 2
        :return: none
        """
        def _custom_format(x):
            """
            Custom formatting for the confusion matrix, if an entry in the confusion matrix is a float number,
            it shows as with .2f precision.
            :param x:
            :return:
            """
            if x == 0:
                return '0'
            else:
                return f'{x:.2f}'

        def _plot_cm(cm, class_names: list, level: str, idx: int, normalized: bool = False) -> None:
            norm_suffix = '_normalized' if normalized else '_counts'
            title_suffix = ' (Normalized)' if normalized else ''

            # Set up annotations based on whether it's normalized or not
            if normalized:
                annot_data = np.array([[_custom_format(val) for val in row] for row in cm])
                fmt = ''
            else:
                annot_data = cm.astype(int)  # Convert to integer format for non-normalized matrix
                fmt = 'd'

            # Create plot
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm, annot=annot_data, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix {level}{title_suffix}')

            # Save CM
            cm_save_path = os.path.join(self.result_save_dir, f'confusion_matrix_{level}{norm_suffix}_cv{idx}.npy')
            np.save(cm_save_path, cm)

            # Log cm numpy array as artifact in wandb
            artifact = wandb.Artifact("cm_numpy_array", type="data")
            artifact.add_file(cm_save_path)
            wandb.log_artifact(artifact)

            # Save the heatmap of CM
            save_path = os.path.join(self.result_save_dir, f'confusion_matrix_{level}{norm_suffix}_cv{idx}.png')
            plt.tight_layout()

            # Log CM to Wandb
            wandb.log({"Confusion Matrix": wandb.Image(plt)})
            plt.savefig(save_path)
            plt.close()  # Close figure to free memory
            # logging.info(f'Confusion matrix {level} plot saved.')

        cv_id = self.cv_id
        cls_desc = class_names_l3
        conf_mat = self.metrics['metrics_l3']['confusion_matrix']
        _plot_cm(conf_mat, cls_desc, 'l3', cv_id, normalized=False)

        # Calculate and save prediction-normalized confusion matrix
        row_sums = conf_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 # Deal with the situation where there is no items in a row of cm
        conf_mat_normalized = conf_mat / row_sums
        _plot_cm(conf_mat_normalized, cls_desc, 'l3', cv_id, normalized=True)

        if self.args.get('l2_metrics', False):
            cls_desc = class_names_l2
            conf_mat = self.metrics['metrics_l2']['confusion_matrix']
            _plot_cm(conf_mat, cls_desc, 'l2', cv_id, normalized=False)

            # Calculate and save prediction-normalized confusion matrix
            row_sums = conf_mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1 # Deal with the situation where there is no items in a row of cm
            conf_mat_normalized = conf_mat / row_sums
            _plot_cm(conf_mat_normalized, cls_desc, 'l2', cv_id, normalized=True)