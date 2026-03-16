import copy
import torch
import torch.nn as nn
import numpy as np
import os
import logging
import wandb
from utils import REASSIGN_LABEL_NAME_L3, REASSIGN_NAME_LABEL_L3L2
from utils.train_utils import (
    get_model_output, scheduler_dict, AverageMeter, l3_to_l2,
    feat_extraction, feat_reduction, draw_latent, UmapLearning)
from .losses import SupConLoss
from methods.core_trainer import BaseTrainer
from sklearn.manifold import TSNE


class SupConTrainerOld:
    def __init__(self, model, classifier, train_loader, val_loader, cv_id, args: dict):
        """
        Trainer for model training and validation
        :param model: model for pretraining, consisting of an encoder and projection head.
        :param classifier: Classifier for the encoder pretrained on SupCon. For downstream task.
        :param train_loader: training set dataloader
        :param val_loader: validation set dataloader
        :param args: arguments for training
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        if args['supcon_conf']['pretrain']:
            self.classifier = None
        else:
            self.classifier = classifier
            self.classifier.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.cv_id = cv_id

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

        if self.args['supcon_conf']['pretrain']:
            model = self.model
        else:
            model = self.classifier

        if self.args['optimiser']['type'] == 'sgd':
            momentum = self.args['optimiser'].get('momentum', 0.9)
            optimiser = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                        momentum=momentum, weight_decay=weight_decay)
        elif self.args['optimiser']['type'] == 'adam':
            optimiser = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
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

    def train(self):
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
            # val_loss, val_top1_acc, val_top1_acc_l2 = self._validate()
            train_losses.append(train_loss)

            # Record the losses
            logging.info(f'\nCV-{cv_id}: Epoch [{epoch + 1}/{num_epochs}] - '
                         f'Avg Training Loss: {train_loss:.4f}\n')
            wandb.log({f"CV-{cv_id}: Train/Loss": train_loss, "epoch": epoch + 1})

            # Save checkpoint periodically
            save_checkpoint = self.args.get('checkpoint', {}).get('save_checkpoint', False)
            if save_checkpoint:
                if (epoch + 1) % self.save_freq == 0 or (epoch + 1 == num_epochs):
                    self._save_checkpoint(epoch)

        return self.model

    def _train_one_epoch(self, epoch) -> float:
        """
        Train the model with SupCon for one epoch.
        :return: Average training loss for the epoch
        """
        args = self.args
        model = self.model
        model.train()

        temperature = args['supcon_conf'].get('temp', 0.1)
        criterion = SupConLoss(temperature=temperature).to(self.device)

        # running_loss = 0.0
        # total_loss = 0.0
        losses = AverageMeter()

        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = torch.cat([images[0], images[1]], dim=0)
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            bsz = labels.shape[0]

            self.optimiser.zero_grad()

            features = get_model_output(model, images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            loss = criterion(features, labels)

            loss.backward()
            self.optimiser.step()

            losses.update(loss.item(), bsz)

            # show running loss every 24 batches
            if (batch_idx + 1) % 24 == 0:
                print(f'Epoch [{epoch + 1}] Batch [{batch_idx + 1}/{len(self.train_loader)}]: '
                      f'Running Loss: {losses.val:.4f}')
        return losses.avg

    def train_classifier(self, return_best_classifier: bool = True):
        """
                The main training loop.
                :return: Trained model and tracked losses
                """
        cv_id = self.cv_id
        num_epochs = self.args['num_epochs']
        train_losses = []
        val_losses = []
        best_classifier = None

        for epoch in range(num_epochs):
            print(f'\nCV-{cv_id}: Epoch {epoch + 1}/{num_epochs}')
            train_loss = self._train_classifier_one_epoch(epoch)
            val_loss, val_top1_acc, val_top1_acc_l2 = self._validate_classifier()

            # Use scheduler for the optimiser
            if self.scheduler:
                if self.args['scheduler']['type'] == 'reduced':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                verbose = self.args['scheduler'].get('verbose', False)
                if verbose:
                    print(f"Epoch [{epoch + 1}] Learning Rate: {self.scheduler.get_last_lr()[0]}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Record the losses
            logging.info(f'\nCV-{cv_id}: Epoch [{epoch + 1}/{num_epochs}] - '
                         f'Avg Training Loss: {train_loss:.4f} | Avg Validation Loss: {val_loss:.4f} | '
                         f'Valid Top1 Acc: {val_top1_acc:.4f}\n')
            wandb.log({f"CV-{cv_id}: Train/Loss": train_loss, "epoch": epoch + 1})
            wandb.log({f"CV-{cv_id}: Valid/Loss": val_loss, "epoch": epoch + 1})
            wandb.log({f"CV-{cv_id}: Valid/Top1_Acc": val_top1_acc, "epoch": epoch + 1})
            if self.args.get('verbose', False):
                wandb.log({f"CV-{cv_id}: Valid/Top1_Acc_L2": val_top1_acc_l2, "epoch": epoch + 1})

            # Save the model with the best overall accuracy
            if self.best_val_top1_acc < val_top1_acc:
                self.best_val_top1_acc = val_top1_acc
                best_classifier = copy.deepcopy(self.classifier.state_dict())

                # Save the best model
                save_best = self.args.get('early_stopping', {}).get('save_best', False)
                if save_best:
                    self._save_classifier_checkpoint(epoch, is_best=True)

            # Early stopping check
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.patience:
                print('Early stopping triggered. Stopping training.')
                break

            # Save checkpoint periodically
            save_checkpoint = self.args.get('checkpoint', {}).get('save_checkpoint', False)
            if save_checkpoint:
                if (epoch + 1) % self.save_freq == 0 or (epoch + 1 == num_epochs):
                    self._save_classifier_checkpoint(epoch)

        if return_best_classifier:
            self.classifier.load_state_dict(best_classifier)

        return self.model, self.classifier, train_losses, val_losses

    def _train_classifier_one_epoch(self, epoch) -> float:
        """
        Train a classifier for the encoder pretrained on SupCon for one epoch.
        :param epoch: Current iteration
        :return: Average training loss
        """
        model = self.model
        classifier = self.classifier

        model.eval()
        classifier.train()
        losses = AverageMeter()

        criterion = nn.CrossEntropyLoss().to(self.device)

        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            bsz = labels.shape[0]

            self.optimiser.zero_grad()

            with torch.no_grad():
                features = get_model_output(model.encoder, images)
            outputs = classifier(features.detach())
            loss = criterion(outputs, labels)

            loss.backward()
            self.optimiser.step()

            losses.update(loss.item(), bsz)

            if (batch_idx + 1) % 24 == 0:
                print(f'Epoch [{epoch + 1}] Batch [{batch_idx + 1}/{len(self.train_loader)}]: '
                      f'Running Loss: {losses.val:.4f}')
        return losses.avg

    def _validate_classifier(self):
        """
        Validate the classifier trained after SupCon pretraining.
        :return:
        """
        model = self.model
        classifier = self.classifier
        model.eval()
        classifier.eval()

        losses = AverageMeter()
        test_correct, test_total, test_correct_l2 = 0, 0, 0

        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for batch_idx, (images, labels, batch_metadata) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                bsz = labels.shape[0]

                outputs = get_model_output(classifier, model.encoder(images))
                loss = criterion(outputs, labels)

                losses.update(loss.item(), bsz)

                # Top-1 accuracy
                _, predicted = torch.max(outputs, 1)
                correct = predicted.eq(labels).sum()
                test_correct += correct.item()
                test_total += labels.size(0)

                # Top-1 accuracy on L2 labels
                if self.args.get('verbose', False):
                    batch_labels_l2 = batch_metadata['l2_label'].to(self.device)
                    batch_predicted_l2 = []
                    for l3_label in predicted:
                        l3_word_label = REASSIGN_LABEL_NAME_L3[l3_label.item()]
                        l2_label = REASSIGN_NAME_LABEL_L3L2[l3_word_label][1]
                        batch_predicted_l2.append(l2_label)
                    batch_predicted_l2 = torch.tensor(batch_predicted_l2, device=predicted.device)
                    correct_l2 = batch_predicted_l2.eq(batch_labels_l2).sum()

                    test_correct_l2 += correct_l2.item()

        avg_loss = losses.avg
        top1_acc = test_correct / test_total
        top1_acc_l2 = 0.0
        if self.args.get('verbose', False):
            top1_acc_l2 = test_correct_l2 / test_total

        return avg_loss, top1_acc, top1_acc_l2

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
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{epoch + 1}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

    def _save_classifier_checkpoint(self, epoch: int, is_best=False) -> None:
        """
        Save the trained classifier and the SupCon-prerained model, only for the linear training phase.
        :param epoch: linear training epoch
        :param is_best:
        :return:
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
            'epoch': epoch,
        }

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'classifier_best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'Best model saved to {best_path}')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'classifier_checkpoint_{epoch + 1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

class SupConTrainer(BaseTrainer):
    def __init__(self, model, classifier, train_loader, val_loader, train_for_valid_dl, cv_id, args: dict):
        super().__init__(model, train_loader, val_loader, cv_id, args)
        if args['supcon_conf']['pretrain']:
            self.classifier = None
        else:
            self.classifier = classifier
            self.classifier.to(self.device)

        # The training set with test transforms for the visualisation of features
        self.train_for_valid_dl = train_for_valid_dl

        self.optimiser = self._create_optimiser()
        self.scheduler = self._create_scheduler()

    def _create_optimiser(self):
        lr = self.args['optimiser']['lr']
        weight_decay = self.args['optimiser'].get('weight_decay', 0)

        if self.args['supcon_conf']['pretrain']:
            model = self.model
        else:
            model = self.classifier

        if self.args['optimiser']['type'] == 'sgd':
            momentum = self.args['optimiser'].get('momentum', 0.9)
            optimiser = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                        momentum=momentum, weight_decay=weight_decay)
        elif self.args['optimiser']['type'] == 'adam':
            optimiser = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                         weight_decay=weight_decay)
        else:
            raise ValueError(f'Optimiser is not supported.')

        if self.args['verbose']:
            logging.info(f"Optimiser {self.args['optimiser']['type']} successfully created: "
                         f"\n {optimiser.state_dict()}")

        return optimiser

    def train(self, return_best_model: bool = True):
        """
        The pretraining loop for SupCon.
        :param return_best_model: Not used, just to meet the abs form.
        :return: Pretrained model
        """
        cv_id = self.cv_id
        num_epochs = self.args['num_epochs']
        train_losses = []
        val_losses = []
        best_model = None

        for epoch in range(num_epochs):
            print(f'\nCV-{cv_id}: Epoch {epoch + 1}/{num_epochs}')
            train_loss = self._train_one_epoch(epoch)
            # val_loss, val_top1_acc, val_top1_acc_l2 = self._validate()
            # self._validate()
            train_losses.append(train_loss)

            # Record the losses
            logging.info(f'\nCV-{cv_id}: Epoch [{epoch + 1}/{num_epochs}] - '
                         f'Avg Training Loss: {train_loss:.4f}\n')
            wandb.log({f"CV-{cv_id}: Train/Loss": train_loss, "epoch": epoch + 1})

            # Save checkpoint periodically
            save_checkpoint = self.args.get('checkpoint', {}).get('save_checkpoint', False)
            if save_checkpoint:
                if (epoch + 1) % self.save_freq == 0 or (epoch + 1 == num_epochs):
                # if 50 <= (epoch + 1) <= 60 and (epoch + 1) % self.save_freq == 0:
                    self._save_checkpoint(epoch)

        return self.model

    def _train_one_epoch(self, epoch) -> float:
        """
        Train the model with SupCon for one epoch.
        :return: Average training loss for the epoch
        """
        args = self.args
        model = self.model
        model.train()

        temperature = args['supcon_conf'].get('temp', 0.1)
        criterion = SupConLoss(temperature=temperature).to(self.device)

        losses = AverageMeter()

        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = torch.cat([images[0], images[1]], dim=0)
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            bsz = labels.shape[0]

            self.optimiser.zero_grad()

            features = get_model_output(model, images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            loss = criterion(features, labels)

            loss.backward()
            self.optimiser.step()

            losses.update(loss.item(), bsz)

            # show running loss every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}] Batch [{batch_idx + 1}/{len(self.train_loader)}]: '
                      f'Running Loss: {losses.val:.4f}')
        return losses.avg

    def _validate(self):
        """
        Placeholder for the validation in Pretraining.
        :return:
        """
        self.model.eval()
        model = self.model.encoder

        # Extract training and validation features from the encoder.
        train_feats, train_labels, _ = feat_extraction(
            model, dl=self.train_for_valid_dl, desc='Extracting training features', device=self.device
        )
        valid_feats, valid_labels, _ = feat_extraction(
            model, dl=self.val_loader, desc='Extracting validation features', device=self.device
        )

        # Learn on the features using UMAP
        umap_learner = UmapLearning(train_feats, train_labels, valid_feats)
        train_emb, valid_emb = umap_learner.get_emb()

        draw_latent(train_emb, train_labels, fig_name='UMAP-Train', use_l2=self.args['use_l2_label'])
        draw_latent(valid_emb, valid_labels, fig_name='UMAP-Test', use_l2=self.args['use_l2_label'])

    def train_classifier(self, return_best_classifier: bool = True):
        """
        The main training loop.
        :return: Trained model and tracked losses
        """
        cv_id = self.cv_id
        num_epochs = self.args['num_epochs']
        train_losses = []
        val_losses = []
        best_classifier = None

        for epoch in range(num_epochs):
            print(f'\nCV-{cv_id}: Epoch {epoch + 1}/{num_epochs}')
            train_loss = self._train_classifier_one_epoch(epoch)
            val_loss, val_top1_acc, val_top1_acc_l2 = self._validate_classifier()

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
            best_classifier = self._save_classifiers(epoch, val_top1_acc, best_classifier, num_epochs)

        if return_best_classifier:
            self.classifier.load_state_dict(best_classifier)

        return self.model, self.classifier, train_losses, val_losses

    def _train_classifier_one_epoch(self, epoch) -> float:
        """
        Train a classifier for the encoder pretrained on SupCon for one epoch.
        :param epoch: Current iteration
        :return: Average training loss
        """
        model = self.model
        classifier = self.classifier

        model.eval()
        classifier.train()
        losses = AverageMeter()

        criterion = nn.CrossEntropyLoss().to(self.device)

        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            bsz = labels.shape[0]

            self.optimiser.zero_grad()

            with torch.no_grad():
                features = get_model_output(model.encoder, images)
            outputs = classifier(features.detach())
            loss = criterion(outputs, labels)

            loss.backward()
            self.optimiser.step()

            losses.update(loss.item(), bsz)

            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}] Batch [{batch_idx + 1}/{len(self.train_loader)}]: '
                      f'Running Loss: {losses.val:.4f}')
        return losses.avg

    def _validate_classifier(self):
        """
        Validate the classifier trained after SupCon pretraining.
        :return:
        """
        model = self.model
        classifier = self.classifier
        model.eval()
        classifier.eval()

        losses = AverageMeter()
        test_correct, test_total, test_correct_l2 = 0, 0, 0

        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for batch_idx, (images, labels, batch_metadata) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                bsz = labels.shape[0]

                outputs = get_model_output(classifier, model.encoder(images))
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

    def _save_classifier_checkpoint(self, epoch: int, is_best=False) -> None:
        """
        Save the trained classifier and the SupCon-prerained model, only for the linear training phase.
        :param epoch: linear training epoch
        :param is_best:
        :return:
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
            'epoch': epoch,
        }

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'classifier_best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'Best model saved to {best_path}')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'classifier_checkpoint_{epoch + 1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

    def _save_classifiers(self, epoch: int, val_top1_acc: float, best_classifier: dict, num_epochs: int) -> dict:
        """
        Update the best model and save it. Also save checkpoint models. Return possibly updated best_model state_dict.
        :param epoch: used for save checkpoint model
        :param val_top1_acc: used for update the self.best_top1_acc
        :param best_classifier: None if not saved
        """
        # Check if we have a new best top-1 accuracy
        if self.best_val_top1_acc < val_top1_acc:
            self.best_val_top1_acc = val_top1_acc
            best_classifier = copy.deepcopy(self.classifier.state_dict())

            # Save the best model
            save_best = self.args.get('early_stopping', {}).get('save_best', False)
            if save_best:
                self._save_classifier_checkpoint(epoch, is_best=True)
        # Save checkpoint periodically
        save_checkpoint = self.args.get('checkpoint', {}).get('save_checkpoint', False)
        if save_checkpoint:
            if (epoch + 1) % self.save_freq == 0 or (epoch + 1 == num_epochs):
                self._save_classifier_checkpoint(epoch)

        return best_classifier
