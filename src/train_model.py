import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.metrics import confusion_matrix, classification_report

from wells_fargo_dataset import WellsFargoDataset
from models.fcn_model import FCN

TRAIN_DATA = '../data/raw/0173eeb640e7-Challenge+Data+Set+-+Campus+Analytics+2020.xlsx'

# Set Seeds so that experiments are reproduceable
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed=123)
torch.manual_seed(456)
torch.cuda.manual_seed_all(789)

# Print average loss per every x amount of batches
PRINT_LOSS_PER_BATCH_NUMBER = 10


class TrainModel:
    def __init__(self, **args):
        # Model
        self.model = self._init_model(
            args['model'], bn=args['bn'], dropout=args['dropout'])

        # Hyperparameters
        self.epochs = args['epochs']
        self.lr = args['lr']
        self.batch_size = args['batch_size']
        self.criterion = self._init_criterion(args['loss_fn'])
        self.optimizer = self._init_optim(args['optim'])

        # Dataset
        self.data_df = pd.read_excel(TRAIN_DATA)
        self.dataset = self._init_dataset(self.data_df)
        self.train_loader, self.test_loader = self._split_dataset(
            self.dataset, self.batch_size)

        # Others
        self.device = self._init_device(
            cpu=args['cpu'], device=args['device'])
        self.exp_no = args['exp_no']
        self.model_path = os.path.join(
            '../models', str(self.exp_no), 'model.sav')

    def _init_model(self, model_type, bn=True, dropout=True):
        if model_type == 'fcn':
            return FCN(bn=bn, dropout=dropout)

    def _init_dataset(self, data):
        dataset = WellsFargoDataset(data)
        return dataset

    def _split_dataset(self, dataset, batch_size, test_split=0.2, shuffle_dataset=True):
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))
        if shuffle_dataset:
            np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler)

        return train_loader, test_loader

    def _init_criterion(self, criterion_type):
        ''' Loss function  '''
        return nn.BCEWithLogitsLoss()

    def _init_device(self, cpu=False, device=0):
        ''' Initialize device. Device can be CPU or integer specifiying a GPU. '''
        if cpu:
            return torch.device('cpu')
        elif torch.cuda.is_available():
            return torch.device(f'cuda:{device}')

    def _init_optim(self, optim_type):
        ''' Initialize optimizer '''
        if optim_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.lr)
        elif optim_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)

    def get_current_lr(self):
        ''' Get learning rate of optimizer '''
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def get_classification_report(self):
        self.model.to(self.device)
        self.model.eval()

        y_test = []
        y_pred = []
        total = len(self.test_loader)

        with torch.no_grad():
            for _, (inputs, char, targets) in enumerate(tqdm(self.test_loader)):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.model(inputs)
                outputs = torch.sigmoid(outputs)
                outputs = torch.round(outputs)

                y_test.extend(targets.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())

        return classification_report(y_test, y_pred)

    def train(self):
        self.model.to(self.device)
        self.model.train()

        print('Training start!')

        for epoch in range(self.epochs):
            running_loss = 0
            for batch_idx, (inputs, chars, targets) in enumerate(tqdm(self.train_loader)):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                # zero out the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                output = self.model(inputs)
                loss = self.criterion(output, targets)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if batch_idx % PRINT_LOSS_PER_BATCH_NUMBER == PRINT_LOSS_PER_BATCH_NUMBER - 1:    # print every 100 mini-batches
                    print('[EPOCH %d, MINI-BATCH %5d, LR %f] loss: %.5f' %
                          (
                              epoch + 1,
                              batch_idx + 1,
                              self.get_current_lr(),
                              running_loss / PRINT_LOSS_PER_BATCH_NUMBER
                          ))
                    running_loss = 0.0

        # Model Checkpointing
        if epoch % 5 == 4:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.model_path)

        print('Finished training.')
        print('Classification Report:\n', self.get_classification_report())
