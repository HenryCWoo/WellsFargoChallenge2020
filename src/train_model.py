from constants import *

import os
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import *

from wells_fargo_dataset import WellsFargoDataset
from models.dense_no_xc_model import DenseNoXC
from models.dense_ohe_model import DenseOHE
from models.dense_embed_model import DenseEmbed
from models.conv1d_embed_model import Conv1DEmbed

TRAIN_DATA = '../data/raw/0173eeb640e7-Challenge+Data+Set+-+Campus+Analytics+2020.xlsx'

# Set Seeds so that experiments are reproduceable
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed=123)
torch.manual_seed(456)
torch.cuda.manual_seed_all(789)


class TrainModel:
    def __init__(self, **args):
        # Save arguments for reinitializing model during k-fold validation
        self.args = args

        # Others
        self.device = self._init_device(
            cpu=args['cpu'], device=args['device'])
        self.exp_no = args['exp_no']
        self.info_yml = os.path.join(MODEL_DIR, str(self.exp_no), 'info.yml')
        self.model_path = os.path.join(
            MODEL_DIR, str(self.exp_no), 'model.sav')

        # Model
        self.model = self._init_model()

        # Hyperparameters
        self.epochs = args['epochs']
        self.lr = args['lr']
        self.batch_size = args['batch_size']
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optim()
        self.scheduler = self._init_scheduler() if args['scheduler'] else None

        # Dataset
        self.data_df = pd.read_excel(TRAIN_DATA)
        self.dataset = WellsFargoDataset(self.data_df)
        self.complete_trainloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=args['batch_size'], shuffle=True)  # Train loader that contains all training data
        self.kfold_generator = kfold_cross_dataset(
            self.dataset, self.batch_size, k=args['kfold'])  # Generates train and validation loaders for k-fold validation

    def _init_model(self):
        model_type = self.args['model']
        device = self.args['device']
        bn = self.args['bn']
        dropout = self.args['dropout']
        dropout_rate = self.args['dropout_rate']

        if model_type == 'dense_no_xc':
            return DenseNoXC(device, bn=bn, dropout=dropout, dropout_rate=dropout_rate)
        elif model_type == 'dense_ohe':
            return DenseOHE(device, bn=bn, dropout=dropout, dropout_rate=dropout_rate)
        elif model_type == 'dense_embed':
            return DenseEmbed(device, bn=bn, dropout=dropout, dropout_rate=dropout_rate)
        elif model_type == 'conv1d_embed':
            return Conv1DEmbed(device, bn=bn, dropout=dropout, dropout_rate=dropout_rate)
        else:
            raise NotImplementedError(
                f'Model type: {model_type} not found. Refer to src.models for models.')

    def _init_criterion(self):
        ''' Loss function  '''
        criterion_type = self.args['loss_fn']
        # return nn.CosineEmbeddingLoss()
        return nn.BCEWithLogitsLoss()

    def _init_device(self, cpu=False, device=0):
        ''' Initialize device. Device can be CPU or integer specifiying a GPU. '''
        if cpu:
            return torch.device('cpu')
        elif torch.cuda.is_available():
            return torch.device(f'cuda:{device}')

    def _init_optim(self):
        ''' Initialize optimizer '''
        optim_type = self.args['optim']

        if optim_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.lr)
        elif optim_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)

    def _init_scheduler(self):
        # Scheduler flag not set
        if not self.args['scheduler']:
            return None

        step_size = self.args['step_size']
        gamma = self.args['gamma']

        return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def _reset_train_session(self):
        self.model = self._init_model()
        self.optimizer = self._init_optim()
        self.scheduler = self._init_scheduler()

    def _average_cls_reports(self, cls_reports):
        if not cls_reports:
            return None

        if len(cls_reports) == 1:
            return cls_reports[0]

        result = cls_reports[0]
        for cls_report in cls_reports[1:]:
            for k, v in cls_report.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        result[k][k2] += v2
                else:
                    result[k] += v

        for k, v in result.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    result[k][k2] /= len(cls_reports)
            else:
                result[k] /= len(cls_reports)

        return result

    def train(self):
        print('k-Fold Cross Validation start.')
        # Collect classification reports of each k-Fold
        cls_reports = []
        for fold_no, (train_loader, valid_loader) in enumerate(self.kfold_generator):
            # Reset optimizer, scheduler, model before training next fold
            # (Resets momentum, learning rate schedule, model weights, etc...)
            self._reset_train_session()
            self.model.to(self.device)
            self.model.train()

            for epoch in range(self.epochs):
                running_loss = 0
                for batch_idx, (inputs, xc, targets) in enumerate(tqdm(train_loader)):
                    inputs, xc, targets = inputs.to(self.device), xc.to(
                        self.device), targets.to(self.device)

                    # zero out the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    output = self.model(inputs, xc)
                    loss = self.criterion(output, targets)

                    # Store loss for logging
                    running_loss += loss.item()

                    loss.backward()
                    self.optimizer.step()

                # print statistics
                print('[k-Fold %d] Epoch %d, LR %.8f, loss: %.8f' % (
                    fold_no + 1,
                    epoch + 1,
                    get_current_lr(self.optimizer),
                    running_loss / self.batch_size
                ))
                running_loss = 0.0

                # Learning Rate Scheduler
                if self.scheduler:
                    self.scheduler.step()

            # Save Classification Report in info.yml
            cls_report = get_classification_report(
                self.device, self.model, valid_loader, output_dict=True)
            cls_reports.append(cls_report)

            print(
                f'\n[Classification Report for K-fold {fold_no + 1}]\n',
                get_classification_report(
                    self.device, self.model, valid_loader, output_dict=False),
                '\n',
            )

        avg_cls_report = self._average_cls_reports(cls_reports)
        with open(self.info_yml, 'r+') as f:
            info_yml = yaml.safe_load(f)
            info_yml.update(
                {'cls_report': avg_cls_report})
            yaml.safe_dump(info_yml, f)

        print('Finished k-Fold Cross Validation.')

        # Build final model
        for epoch in range(self.epochs):
            running_loss = 0
            for batch_idx, (inputs, xc, targets) in enumerate(tqdm(self.complete_trainloader)):
                inputs, xc, targets = inputs.to(self.device), xc.to(
                    self.device), targets.to(self.device)

                # zero out the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                output = self.model(inputs, xc)
                loss = self.criterion(output, targets)

                # Store loss for logging
                running_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            # print statistics
            print('[FINAL MODEL] Epoch %d, LR %.8f, loss: %.8f' % (
                epoch + 1,
                get_current_lr(self.optimizer),
                running_loss / self.batch_size
            ))
            running_loss = 0.0

            # Learning Rate Scheduler
            if self.scheduler:
                self.scheduler.step()

        # Save Final Model
        torch.save({'model_state_dict': self.model.state_dict()},
                   self.model_path)
        print(
            f"AVERAGE WEIGHTED F-1 SCORE: {avg_cls_report['weighted avg']['f1-score']}")
