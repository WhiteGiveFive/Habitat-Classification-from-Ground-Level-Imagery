from models import create_model, SupConEncoder, LinearClassifier
from methods.trainer import Trainer
from methods.supcon.supcon_trainer import SupConTrainer
from methods.evaluator import Evaluator
from torchinfo import summary
import os
import torch
from utils.model_utils import load_model_params

class Executor:
    """
    Executing the learning method using trainer and evaluator.
    """
    def __init__(self, args):
        """
        Initialising the Executor.
        :param args: Overall configurations
        """
        self.args = args

    def cross_valid(self, cv_dataloader, num_classes, class_names_l2, class_names_l3, results_dir):
        """
        Perform cross validation on the model
        :param cv_dataloader: Dataloaders for cross validation
        :param num_classes: Number of classes, a tuple consists of (number of classes L3, number of classes L2)
        :param class_names_l2: Names of classes L2.
        :param class_names_l3: Names of classes L3.
        :param results_dir: Directory where results are saved.
        :return:
        """
        cv_test_metrics = []
        args = self.args

        model_name = args['model']['name']
        for cv_id in range(len(cv_dataloader.trainvalid_dls)):
            train_loader, val_loader = cv_dataloader.get_dataloaders(cv_id)

            model = create_model(model_name, cv_id, args['model'])
            trainer = Trainer(model, train_loader, val_loader, cv_id=cv_id, args=args['training'])

            trained_model, train_losses, val_losses = trainer.train(return_best_model=True)

            evaluator = Evaluator(trained_model, val_loader, num_classes, results_dir, cv_id=cv_id,
                                  args=args['evaluation'])
            single_test_results = evaluator.test()
            cv_test_metrics.append(single_test_results)

            # Save the misclassified info and confusion matrix for the current cross validation.
            evaluator.save_misclassified()
            evaluator.save_cm(class_names_l3=class_names_l3, class_names_l2=class_names_l2)
            print("=" * 90)
        return cv_test_metrics

    def train_test(self, train_test_dls, num_classes, class_names_l2, class_names_l3, results_dir):
        """
        Perform the training and test.
        :param train_test_dls: Dataloaders for training and test
        :param num_classes: Number of classes, a tuple consists of (number of classes L3, number of classes L2)
        :param class_names_l2:
        :param class_names_l3:
        :param results_dir: Directory where results are saved.
        :return:
        """
        args = self.args
        model_name = args['model']['name']

        train_dl, test_dl = train_test_dls.get_dataloaders()

        # Create the model, cv_id is set to 0 to print out the model info
        model = create_model(model_name, 0, args['model'])

        # Training
        trainer = Trainer(model, train_dl, test_dl, cv_id='test', args=args['training'])
        trained_model, train_losses, val_losses = trainer.train(return_best_model=True)

        # Perform a test to record the confusion matrix and misclassification
        evaluator = Evaluator(trained_model, test_dl, num_classes, results_dir, cv_id='test', args=args['evaluation'])
        single_test_results = evaluator.test()
        evaluator.save_misclassified()
        evaluator.save_cm(class_names_l3=class_names_l3, class_names_l2=class_names_l2)
        print("=" * 90)
        # Return the results as a list, so that we can uniform the metrics calculation in the main file
        return [single_test_results]

class SupConExecutor:
    """
    Executing the Supervised Contrastive Learning using trainer and evaluator.
    """
    def __init__(self, args):
        """
        Initialising the Executor.
        :param args: Overall configurations
        """
        self.args = args

    def _set_supcon_args(self):
        args = self.args
        train_args = args['training']
        model_args = args['model']
        model_name = model_args['name']
        model_input_size = model_args['input_size']
        return train_args, model_args, model_name, model_input_size

    def cross_valid(self, cv_dataloader):
        """
        Perform cross validation on the model
        :param cv_dataloader: Dataloaders for cross validation
        :return:[]
        """
        cv_test_metrics = []
        train_args, model_args, model_name, model_input_size = self._set_supcon_args()
        classifier = None
        num_cv = len(cv_dataloader.trainvalid_dls)

        for cv_id in range(num_cv):
            train_loader, val_loader = cv_dataloader.get_dataloaders(cv_id)

            model = SupConEncoder(model_name, cv_id, model_args)
            summary(model, input_size=(8, 3, model_input_size, model_input_size))

            trainer = SupConTrainer(model, classifier, train_loader, val_loader, train_for_valid_dl=None, cv_id=cv_id, args=train_args)

            pretrained_model = trainer.train()

        return cv_test_metrics

    def train_test(self, train_test_dls):
        """
        Perform the training and test.
        :param train_test_dls: Dataloaders for training and test
        :return:
        """
        train_args, model_args, model_name, model_input_size= self._set_supcon_args()
        classifier = None

        # train_for_valid is set to True for SupCon pretrain to generate the training set with test transforms
        train_dl, test_dl, train_for_valid_dl = train_test_dls.get_dataloaders(train_for_valid=True)

        # Create the model, cv_id is set to 0 to print out the model info
        model = SupConEncoder(model_name, 0, model_args)
        summary(model, input_size=(8, 3, model_input_size, model_input_size))

        if model_args.get('source', False):
            load_model_params(model, model_args)
        # Pretraining
        print('==========>Starting SupCon pretraining...<==========')
        trainer = SupConTrainer(model, classifier, train_dl, test_dl, train_for_valid_dl, cv_id='test', args=train_args)
        pretrained_model = trainer.train()

        # Return the results as a list, so that we can uniform the metrics calculation in the main file
        return []

    def train_test_classifier(self, train_test_dls, num_classes, class_names_l2, class_names_l3, results_dir):
        """
        Train and test the linear classifier for SupCon encoder on the training and test sets.
        :param train_test_dls:
        :param num_classes: Number of classes, a tuple consists of (number of classes L3, number of classes L2)
        :param class_names_l2:
        :param class_names_l3:
        :param results_dir: Directory where results are saved.
        :return: 
        """
        train_args, model_args, model_name, model_input_size = self._set_supcon_args()

        train_dl, test_dl = train_test_dls.get_dataloaders()
        model = SupConEncoder(model_name, 0, model_args)
        summary(model, input_size=(8, 3, model_input_size, model_input_size))
        classifier = LinearClassifier(model_args)

        # Load the pretrained model
        if not train_args['supcon_conf']['pretrain']:
            loaded_model_path = os.path.join(train_args['supcon_conf']['prt_dir'], train_args['supcon_conf']['prt_filename'])
            checkpoint = torch.load(loaded_model_path, map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"==========>Pretrained model loaded from {loaded_model_path}<==========")

        trainer = SupConTrainer(model, classifier, train_dl, test_dl, train_for_valid_dl=None, cv_id='test', args=train_args)

        trained_model, trained_classifier, train_losses, val_losses = trainer.train_classifier(return_best_classifier=True)

        # Perform a test to record the confusion matrix and misclassification
        evaluator = Evaluator(trained_model, test_dl, num_classes, results_dir, cv_id='test', args=self.args['evaluation'])
        single_test_results = evaluator.test_classifier(trained_classifier)
        evaluator.save_misclassified()
        evaluator.save_cm(class_names_l3=class_names_l3, class_names_l2=class_names_l2)
        print("=" * 90)
        # Return the results as a list, so that we can uniform the metrics calculation in the main file
        return [single_test_results]
