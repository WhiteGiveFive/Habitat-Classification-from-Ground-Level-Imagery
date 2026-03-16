import copy
import torch
import torch.nn as nn
import numpy as np
import os
import logging
import wandb
from utils.train_utils import get_model_output, scheduler_dict, AverageMeter, l3_to_l2
from methods.core_trainer import BaseTrainer


class Trainer:
    def __init__(self, model, train_loader, val_loader, cv_id, args: dict):
        """
        Trainer for model training and validation
        :param model: model to train
        :param train_loader: training set dataloader
        :param val_loader: validation set dataloader
        :param args: arguments for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.cv_id = cv_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create optimizer and scheduler
        self.optimiser = self._create_optimiser()
        self.scheduler = self._create_scheduler()

        # Early stopping setup
        self.patience = args.get('early_stopping', {}).get('patience', 10)
        self.min_delta = args.get('early_stopping', {}).get('min_delta', 0.001)
        self.early_stopping_counter = 0
        self.best_val_loss = np.inf
        self.best_val_top1_acc = 0.0

        # Checkpoint setup
        self.checkpoint_dir = args['checkpoint']['save_path'] # save path is created in the main.py
        self.save_freq = args.get('checkpoint', {}).get('save_freq', 5)

        self.model.to(self.device)

    def _create_optimiser(self):
        lr = self.args['optimiser']['lr']
        weight_decay = self.args['optimiser'].get('weight_decay', 0)

        if self.args['optimiser']['type'] == 'sgd':
            momentum = self.args['optimiser'].get('momentum', 0.9)
            optimiser = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                        momentum=momentum, weight_decay=weight_decay)
        elif self.args['optimiser']['type'] == 'adam':
            optimiser = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                         weight_decay=weight_decay)
        else:
            raise ValueError(f'Optimiser is not supported.')

        if self.args['verbose']:
            logging.info(f"Optimiser {self.args['optimiser']['type']} successfully created: "
                         f"\n {optimiser.state_dict()}")

        return optimiser

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
                raise KeyError(f"Invalid scheduler type. Supported types are: {supported_scheduler}.")

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
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.patience:
            print('Early stopping triggered. Stopping training.')
            return True
        return False

    def _save_models(self, epoch: int, val_top1_acc: float, best_model: dict, num_epochs: int) -> dict:
        """
        Update the best model and save it. Also save checkpoint models. Return possibly updated best_model state_dict.
        :param epoch: used for save checkpoint model
        :param val_top1_acc: used for update the self.best_top1_acc
        :param best_model: None if not saved
        """
        # Check if we have a new best top-1 accuracy
        if self.best_val_top1_acc < val_top1_acc:
            self.best_val_top1_acc = val_top1_acc
            best_model = copy.deepcopy(self.model.state_dict())

            # Save best model if configured
            save_best = self.args.get('early_stopping', {}).get('save_best', False)
            if save_best:
                self._save_checkpoint(epoch, is_best=True)

        # Periodic checkpoint saving
        save_checkpoint = self.args.get('checkpoint', {}).get('save_checkpoint', False)
        if save_checkpoint:
            if (epoch + 1) % self.save_freq == 0 or (epoch + 1 == num_epochs):
                self._save_checkpoint(epoch)

        return best_model

    def train(self, return_best_model: bool = True):
        """
        The main training loop.
        :return: Trained model and tracked losses
        """
        cv_id = self.cv_id
        num_epochs = self.args['num_epochs']
        train_losses = []
        val_losses = []
        best_model = None

        for epoch in range(num_epochs):
            print(f'\nCV-{cv_id}: Epoch {epoch + 1}/{num_epochs}')
            train_loss = self._train_one_epoch(epoch)
            val_loss, val_top1_acc, val_top1_acc_l2 = self._validate()

            # Update the scheduler
            self._scheduler_update(epoch, val_loss)

            # Record performance
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            self._wandb_record(epoch + 1, train_loss, val_loss, val_top1_acc, val_top1_acc_l2)

            # Early stopping check
            if self._early_stopping_check(val_loss):
                break

            # Save best model and checkpoint models if set Ture
            best_model = self._save_models(epoch, val_top1_acc, best_model, num_epochs)

        if return_best_model:
            self.model.load_state_dict(best_model)

        return self.model, train_losses, val_losses

    def _train_one_epoch(self, epoch) -> float:
        """
        Train the model for one epoch.
        :return: Average training loss for the epoch
        """
        model = self.model
        model.train()

        criterion = nn.CrossEntropyLoss().to(self.device)

        losses = AverageMeter()

        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            # The batched data could be a list if edge augmentation is applied.
            if isinstance(images, list):
                images = torch.cat([images[0], images[1]], dim=0)
                labels = torch.cat([labels, labels], dim=0)
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            bsz = labels.shape[0]

            self.optimiser.zero_grad()

            outputs = get_model_output(model, images)
            loss = criterion(outputs, labels)

            loss.backward()
            self.optimiser.step()

            losses.update(loss.item(), bsz)

            # show running loss every 48 batches
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}] Batch [{batch_idx + 1}/{len(self.train_loader)}]: '
                      f'Running Loss: {losses.val:.4f}')

        return losses.avg

    def _validate(self):
        """
        Validate the
        :return:
        """
        model = self.model
        model.eval()
        losses = AverageMeter()

        test_correct, test_total, test_correct_l2 = 0, 0, 0

        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for batch_idx, (images, labels, batch_metadata) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                bsz = labels.shape[0]

                outputs = get_model_output(model, images)
                loss = criterion(outputs, labels)

                losses.update(loss.item(), bsz)

                # Top-1 accuracy
                _, predicted = torch.max(outputs, 1)
                correct = predicted.eq(labels).sum()
                test_correct += correct.item()
                test_total += labels.size(0)

                # Top-1 accuracy on L2 labels
                if self.args.get('verbose', False):
                    correct_l2 = l3_to_l2(predicted, batch_metadata, self.device)
                    test_correct_l2 += correct_l2.item()

        top1_acc = test_correct / test_total
        top1_acc_l2 = 0.0
        if self.args.get('verbose', False):
            top1_acc_l2 = test_correct_l2 / test_total

        return losses.avg, top1_acc, top1_acc_l2

    def _save_checkpoint(self, epoch: int, is_best=False) -> None:
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

class TrainerABS(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, cv_id, args: dict):
        """
        Trainer for model training and validation
        :param model: model to train
        :param train_loader: training set dataloader
        :param val_loader: validation set dataloader
        :param args: arguments for training
        """
        super().__init__(model, train_loader, val_loader, cv_id, args)

        # Create optimizer and scheduler
        self.optimiser = self._create_optimiser()
        self.scheduler = self._create_scheduler()

    def _create_optimiser(self):
        lr = self.args['optimiser']['lr']
        weight_decay = self.args['optimiser'].get('weight_decay', 0)

        if self.args['optimiser']['type'] == 'sgd':
            momentum = self.args['optimiser'].get('momentum', 0.9)
            optimiser = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                        momentum=momentum, weight_decay=weight_decay)
        elif self.args['optimiser']['type'] == 'adam':
            optimiser = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                         weight_decay=weight_decay)
        else:
            raise ValueError(f'Optimiser is not supported.')

        if self.args['verbose']:
            logging.info(f"Optimiser {self.args['optimiser']['type']} successfully created: "
                         f"\n {optimiser.state_dict()}")

        return optimiser

    def _save_models(self, epoch: int, val_top1_acc: float, best_model: dict, num_epochs: int) -> dict:
        """
        Update the best model and save it. Also save checkpoint models. Return possibly updated best_model state_dict.
        :param epoch: used for save checkpoint model
        :param val_top1_acc: used for update the self.best_top1_acc
        :param best_model: None if not saved
        """
        # Check if we have a new best top-1 accuracy
        if self.best_val_top1_acc < val_top1_acc:
            self.best_val_top1_acc = val_top1_acc
            best_model = copy.deepcopy(self.model.state_dict())

            # Save best model if configured
            save_best = self.args.get('early_stopping', {}).get('save_best', False)
            if save_best:
                self._save_checkpoint(epoch, is_best=True)

        # Periodic checkpoint saving
        save_checkpoint = self.args.get('checkpoint', {}).get('save_checkpoint', False)
        if save_checkpoint:
            if (epoch + 1) % self.save_freq == 0 or (epoch + 1 == num_epochs):
                self._save_checkpoint(epoch)

        return best_model

    def train(self, return_best_model: bool = True):
        """
        The main training loop.
        :return: Trained model and tracked losses
        """
        cv_id = self.cv_id
        num_epochs = self.args['num_epochs']
        train_losses = []
        val_losses = []
        best_model = None

        for epoch in range(num_epochs):
            print(f'\nCV-{cv_id}: Epoch {epoch + 1}/{num_epochs}')
            train_loss = self._train_one_epoch(epoch)
            val_loss, val_top1_acc, val_top1_acc_l2 = self._validate()

            # Update the scheduler
            self._scheduler_update(epoch, val_loss)

            # Record performance
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            self._wandb_record(epoch + 1, train_loss, val_loss, val_top1_acc, val_top1_acc_l2)

            # Early stopping check
            if self._early_stopping_check(val_loss):
                break

            # Save best model and checkpoint models if set Ture
            best_model = self._save_models(epoch, val_top1_acc, best_model, num_epochs)

        if return_best_model:
            self.model.load_state_dict(best_model)

        return self.model, train_losses, val_losses

    def _train_one_epoch(self, epoch) -> float:
        """
        Train the model for one epoch.
        :return: Average training loss for the epoch
        """
        model = self.model
        model.train()

        criterion = nn.CrossEntropyLoss().to(self.device)

        losses = AverageMeter()

        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            bsz = labels.shape[0]

            self.optimiser.zero_grad()

            outputs = get_model_output(model, images)
            loss = criterion(outputs, labels)

            loss.backward()
            self.optimiser.step()

            losses.update(loss.item(), bsz)

            # show running loss every 48 batches
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}] Batch [{batch_idx + 1}/{len(self.train_loader)}]: '
                      f'Running Loss: {losses.val:.4f}')

        return losses.avg

    def _validate(self):
        """
        Validate the
        :return:
        """
        model = self.model
        model.eval()
        losses = AverageMeter()

        test_correct, test_total, test_correct_l2 = 0, 0, 0

        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for batch_idx, (images, labels, batch_metadata) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                bsz = labels.shape[0]

                outputs = get_model_output(model, images)
                loss = criterion(outputs, labels)

                losses.update(loss.item(), bsz)

                # Top-1 accuracy
                _, predicted = torch.max(outputs, 1)
                correct = predicted.eq(labels).sum()
                test_correct += correct.item()
                test_total += labels.size(0)

                # Top-1 accuracy on L2 labels
                if self.args.get('verbose', False):
                    correct_l2 = l3_to_l2(predicted, batch_metadata, self.device)
                    test_correct_l2 += correct_l2.item()

        top1_acc = test_correct / test_total
        top1_acc_l2 = 0.0
        if self.args.get('verbose', False):
            top1_acc_l2 = test_correct_l2 / test_total

        return losses.avg, top1_acc, top1_acc_l2
