import argparse
import logging
import os
from config.config_parser import load_config
import wandb
import random
import torch
import numpy as np

from data.dataloader import CrossValidDataloaders, TrainTestDataLoaders
from utils import REASSIGN_LABEL_NAME_L3, NAME_LABEL_L2, REASSIGN_NAME_LABEL_L3
from utils.main_utils import avg_performance, log_model_performance

from methods.executor import SupConExecutor


def main():
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Load configuration and define the experiment name
    parser = argparse.ArgumentParser(description='AI Hab training and evaluation')
    parser.add_argument('--config', type=str, default='config/default_config.yaml', help='Path to config file')
    parser.add_argument('--run-id', type=str, required=True, default='example_run', help='Run ID')
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(config)

    experiment_name = config['model']['name']+(f"-epoch{config['training']['num_epochs']}"
                                               f"-{config['training']['optimiser']['type']}"
                                               f"-lr{config['training']['optimiser']['lr']}"
                                               f"-seed{config['seed']}-"
                                               f"{args.run_id}")
    logger.info(experiment_name)

    # Wandb initialise
    wandb.init(
        project='aihab-p2-contrastive',
        name=experiment_name,
        config=config)

    # Set up experiment saving dir and folder creation
    experiment_dir = str(os.path.join(config['save_dir'], args.run_id))
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    results_dir = os.path.join(experiment_dir, 'results')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    config['training']['checkpoint']['save_path'] = checkpoint_dir

    # Seed state
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])


    class_names_l3, class_names_l2 = list(REASSIGN_LABEL_NAME_L3.values()), list(NAME_LABEL_L2.keys())
    num_classes = (len(REASSIGN_NAME_LABEL_L3), len(NAME_LABEL_L2))
    config['model']['num_classes'] = len(REASSIGN_NAME_LABEL_L3)

    # Reconfig the number of classes when using L2 label for training
    if config['data'].get('use_l2_label', False):
        config['model']['num_classes'] = len(NAME_LABEL_L2)
        config['evaluation']['l2_metrics'] = False  # Close l2 metric report if training with l2 label.
        class_names_l3 = class_names_l2
        num_classes = (len(NAME_LABEL_L2), len(NAME_LABEL_L2))

    # Set up specific configuration for the SupCon
    train_config = config['training']
    if train_config.get('supcon', False):
        executor = SupConExecutor(config)
    else:
        raise ValueError("supcon should set to True in the configuration.")

    data_config = config['data']
    if train_config['supcon_conf']['pretrain']:
        data_config['preprocessing']['multi_views']['supcon'] = True
    else:
        data_config['preprocessing']['multi_views']['supcon'] = False

    if config.get('cross_valid', False):
        cv_dl = CrossValidDataloaders(data_config)
        test_metrics = executor.cross_valid(cv_dl)
    else:
        train_test_dl = TrainTestDataLoaders(data_config)
        if train_config['supcon_conf']['pretrain']:
            _ = executor.train_test(train_test_dl)
        else:
            test_metrics = executor.train_test_classifier(
                train_test_dl, num_classes, class_names_l2, class_names_l3, results_dir
            )

            avg_metrics = avg_performance(test_metrics)

            log_model_performance(avg_metrics['metrics_l3'], 'l3')
            if config['evaluation'].get('l2_metrics', False):
                log_model_performance(avg_metrics['metrics_l2'], 'l2')


if __name__ == '__main__':
    main()
