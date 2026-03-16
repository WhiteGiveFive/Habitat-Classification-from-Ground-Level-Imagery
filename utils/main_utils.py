import logging
import matplotlib.pyplot as plt
import os
from PIL import Image
import pandas as pd
import numpy as np


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def avg_performance(test_metrics: list) -> dict:
    """
    Averaging the performance in cross validation.
    :param test_metrics: A list of test metrics in the cross validation.
    :return: A dictionary with the average performance, consisting both metrics l3 and metrics l2.
    """
    avg_performance = {
        'metrics_l3': {
            'test_loss': (0.0, 0.0),
            'top1_acc': (0.0, 0.0),
            'top3_acc': (0.0, 0.0),
            'f1_score': (0.0, 0.0),
        },
        'metrics_l2': {
            'test_loss': (0.0, 0.0),
            'top1_acc': (0.0, 0.0),
            'top3_acc': (0.0, 0.0),
            'f1_score': (0.0, 0.0),
        }
    }
    cv_count = 0
    test_loss_l3 = []
    top1_acc_l3 = []
    top3_acc_l3 = []
    f1_score_l3 = []
    top1_acc_l2 = []
    f1_score_l2 = []

    for cv_id, metric in enumerate(test_metrics):
        test_loss_l3.append(metric['metrics_l3']['test_loss'])
        top1_acc_l3.append(metric['metrics_l3']['top1_acc'])
        top3_acc_l3.append(metric['metrics_l3']['top3_acc'])
        f1_score_l3.append(metric['metrics_l3']['f1_score'])

        top1_acc_l2.append(metric['metrics_l2']['top1_acc'])
        f1_score_l2.append(metric['metrics_l2']['f1_score'])

        # avg_performance['metrics_l3']['test_loss'] += metric['metrics_l3']['test_loss']
        #
        # avg_performance['metrics_l3']['top1_acc'] += metric['metrics_l3']['top1_acc']
        # avg_performance['metrics_l2']['top1_acc'] += metric['metrics_l2']['top1_acc']
        #
        # avg_performance['metrics_l3']['top3_acc'] += metric['metrics_l3']['top3_acc']
        #
        # avg_performance['metrics_l3']['f1_score'] += metric['metrics_l3']['f1_score']
        # avg_performance['metrics_l2']['f1_score'] += metric['metrics_l2']['f1_score']

        # cv_count += 1

    avg_performance['metrics_l3']['test_loss'] = (np.array(test_loss_l3).mean(), np.array(test_loss_l3).std())
    avg_performance['metrics_l3']['top1_acc'] = (np.array(top1_acc_l3).mean(), np.array(top1_acc_l3).std())
    avg_performance['metrics_l3']['top3_acc'] = (np.array(top3_acc_l3).mean(), np.array(top3_acc_l3).std())
    avg_performance['metrics_l3']['f1_score'] = (np.array(f1_score_l3).mean(), np.array(f1_score_l3).std())

    avg_performance['metrics_l2']['top1_acc'] = (np.array(top1_acc_l2).mean(), np.array(top1_acc_l2).std())
    avg_performance['metrics_l2']['f1_score_l2'] = (np.array(f1_score_l2).mean(), np.array(f1_score_l2).std())

    # avg_performance['metrics_l3'] = {key: value / cv_count for key, value in avg_performance['metrics_l3'].items()}
    # avg_performance['metrics_l2'] = {key: value / cv_count for key, value in avg_performance['metrics_l2'].items()}

    return avg_performance

def log_model_performance(test_metrics: dict, label_level: str):
    """
    Log the model performance with formatting to separate the information.
    :param test_metrics: Dictionary containing the test loss, accuracies, and F1 score.
    :param label_level: Label level, l2 or l3
    """
    logging.info("-" * 60)
    logging.info(f"| {'Metric_'+label_level:<15} | {'CV_Mean':<20} | {'CV_Std':<20} |")
    logging.info("-" * 60)

    for metric, values in test_metrics.items():
        mean, std_dev = values
        logging.info(f"| {metric:<15} | {mean:<20.4f} | {std_dev:<20.4f} |")

    logging.info("-" * 60)

def display_misclassification(misclassification_file_path, image_dir):
    """
    Display the misclassified images
    :param misclassification_file: csv file with misclassified image info
    :param image_dir: the folder where the images are stored
    :return:
    """
    df = pd.read_csv(misclassification_file_path)

    # Sort the dataframe by the ground truth label for grouping
    df.sort_values(by=['ground_truth_num_label'], inplace=True)

    unique_labels = df['ground_truth_num_label'].unique()

    # Set up the grid
    n_cols = 5
    n_rows = len(unique_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    if n_rows == 1:
        axes = [axes]

    for row_idx, label in enumerate(unique_labels):
        # Subset the dataframe to get all misclassified images with the same ground
        label_df = df[df['ground_truth_num_label'] == label]
        plot_word_label = label_df['ground_truth_word_label'].iloc[0]  # Ground truth word for this row

        # Annotate the row with the ground truth word label
        fig.text(0.5, (n_rows - row_idx - 0.5) / n_rows, f"Ground Truth: {plot_word_label}",
                 ha='center', fontsize=14, fontweight='bold')

        # Display images in the row (fill with blank if less than n_cols)
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx] if n_rows > 1 else axes[col_idx]
            ax.axis('off')

            if col_idx < len(label_df):
                # Load the image
                img_file_name = label_df.iloc[col_idx]['file_name']
                img_path = os.path.join(image_dir, img_file_name)
                img = Image.open(img_path)

                # Show the image
                ax.imshow(img)

                # Annotate with the predicted label
                predicted_label = label_df.iloc[col_idx]['predicted_label']
                ax.set_title(f"Predicted: {predicted_label}", fontsize=12)
    # Adjust layout
    plt.tight_layout()
    plt.show()

def set_nested_value(main_conf: dict, nested_keys: list, updated_value, verbose: bool = False) -> None:
    """
    This function aims to update the main configurations with the sweep configurations
    :param main_conf: The main configurations dictionary.
    :param nested_keys: The chains of the nested keys, the values in the SWEEP_KEY_MAPPING.
    :param updated_value: The new value to update the config. It shall come from the sweep configs.
    :param verbose: If True, prints each update step (default is False).
    :return:
    """
    for key in nested_keys[:-1]:
        main_conf = main_conf.setdefault(key, {})
    main_conf[nested_keys[-1]] = updated_value
    if verbose:
        print(f"Set {nested_keys[-1]} to {updated_value} at {nested_keys}")
