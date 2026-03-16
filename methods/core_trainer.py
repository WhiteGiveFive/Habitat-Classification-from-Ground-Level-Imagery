import abc
import torch
import torch.nn as nn
import numpy as np
import os
import copy
import wandb
import logging

from utils.train_utils import AverageMeter, get_model_output, scheduler_dict

class BaseTrainer(abc.ABC):
    """
    Abstract BaseTrainer that holds shared methods and attributes
    for any training approach. Any methods that differ among trainers
    should be kept abstract or overridden by subclasses.
    """

    def __init__(self, model, train_loader, val_loader, cv_id, args: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.cv_id = cv_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Create optimizer and scheduler
        self.optimiser = None
        self.scheduler = None

        # Early stopping setup
        self.patience = args.get('early_stopping', {}).get('patience', 10)
        self.min_delta = args.get('early_stopping', {}).get('min_delta', 0.001)
        self.early_stopping_counter = 0
        self.best_val_loss = np.inf
        self.best_val_top1_acc = 0.0

        # Checkpoint setup
        self.checkpoint_dir = args['checkpoint']['save_path']
        self.save_freq = args.get('checkpoint', {}).get('save_freq', 5)

    def _create_scheduler(self):
        scheduler = None
        args = self.args
        if args.get('scheduler', False):
            scheduler_name = args['scheduler']['type']

            # Check the scheduler type is supported
            if scheduler_name in scheduler_dict:
                scheduler_class = scheduler_dict[scheduler_name]
            else:
                supported_scheduler = ", ".join(map(str, scheduler_dict.keys()))
                raise KeyError(f"Invalid scheduler {scheduler_name}. Supported types are: {supported_scheduler}.")

            # Create the scheduler
            if scheduler_name == 'cosine':
                scheduler = scheduler_class(self.optimiser, T_max=args['num_epochs'])
            elif scheduler_name == 'reduced':
                factor = args['scheduler']['config']['reduced_factor']
                patience = args['scheduler']['config']['reduced_patience']
                scheduler = scheduler_class(self.optimiser, mode='min', factor=factor, patience=patience)
        return scheduler

    def _scheduler_update(self, epoch: int, val_loss: float):
        """
        Update the scheduler.
        """
        if self.scheduler:
            if self.args['scheduler']['type'] == 'reduced':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            verbose = self.args['scheduler'].get('verbose', False)
            if verbose:
                print(f"Epoch [{epoch + 1}] Learning Rate: {self.scheduler.get_last_lr()[0]}")

    def load_model_params(self, model_params: dict, return_model: bool = False):
        """
        Load parameters for the model
        :param model_params: parameters for the model, not include optimiser for future training
        :param return_model: if return the model
        :return:
        """
        self.model.load_state_dict(model_params)
        if return_model:
            return self.model

    def _early_stopping_check(self, val_loss: float) -> bool:
        """
        Check whether to trigger early stopping based on val_loss.
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.patience:
            print('Early stopping triggered. Stopping training.')
            return True
        return False

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save the model checkpoint.
        :param epoch: Current epoch number.
        :param is_best: If True, save the model as the best model so far.
        :return:
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
            'epoch': epoch,
            'best_val_loss': self.best_val_loss
        }

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'Best model saved to {best_path}')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{epoch + 1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

    def _wandb_record(self, epoch: int, train_loss: float, val_loss: float,
                      val_top1_acc: float, val_top1_acc_l2: float) -> None:
        """
        Log metrics to wandb.
        """
        cv_id = self.cv_id
        logging.info(
            f'\nCV-{cv_id}: Epoch [{epoch}/{self.args["num_epochs"]}] - '
            f'Avg Training Loss: {train_loss:.4f} | Avg Validation Loss: {val_loss:.4f} | '
            f'Valid Top1 Acc: {val_top1_acc:.4f}\n'
        )
        wandb.log({f"CV-{cv_id}: Train/Loss": train_loss, "epoch": epoch})
        wandb.log({f"CV-{cv_id}: Valid/Loss": val_loss, "epoch": epoch})
        wandb.log({f"CV-{cv_id}: Valid/Top1_Acc": val_top1_acc, "epoch": epoch})

        if self.args.get('verbose', False):
            wandb.log({f"CV-{cv_id}: Valid/Top1_Acc_L2": val_top1_acc_l2, "epoch": epoch})

    @abc.abstractmethod
    def _create_optimiser(self):
        """
        Child classes must implement the logic for a single training epoch.
        :return:
        """
        pass

    @abc.abstractmethod
    def train(self, return_best_model: bool = True):
        """
        Child classes must implement the logic for a single training epoch.
        :return:
        """
        pass

    @abc.abstractmethod
    def _train_one_epoch(self, epoch: int) -> float:
        """
        Child classes must implement the logic for a single training epoch.
        Return: average training loss for that epoch.
        """
        pass

    @abc.abstractmethod
    def _validate(self):
        """
        Child classes must implement validation logic.
        Return: (val_loss, val_top1_acc, val_top1_acc_l2)
        """
        pass
