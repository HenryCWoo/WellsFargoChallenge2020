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

TEST_DATA = '../data/raw/d59675225279-Evaluation+Data+Set+-+Campus+Analytics+2020.xlsx'

# Set Seeds so that experiments are reproduceable
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed=123)
torch.manual_seed(456)
torch.cuda.manual_seed_all(789)


class TestModel:
    def __init__(self, exp_no, cpu, device):
        # Others
        self.exp_no = exp_no
        self.device = self._init_device(
            cpu=cpu, device=device)

        self.info_yml = os.path.join(
            EXPERIMENTS_DIR, str(self.exp_no), 'info.yml')
        self.model_path = os.path.join(
            EXPERIMENTS_DIR, str(self.exp_no), 'model.sav')
        self.output_path = os.path.join(
            EXPERIMENTS_DIR, str(self.exp_no), 'output.csv')

        with open(self.info_yml, 'r') as f:
            self.args = yaml.safe_load(f)

        # Model
        self.model = self._init_model()
        self._load_model_weights()

        # Dataset
        self.data_df = pd.read_excel(TEST_DATA)
        self.dataset = WellsFargoDataset(self.data_df, train=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.args['batch_size'], shuffle=False)  # Train loader that contains all training data

    def _init_device(self, cpu=False, device=0):
        ''' Initialize device. Device can be CPU or integer specifiying a GPU. '''
        if cpu:
            return torch.device('cpu')
        elif torch.cuda.is_available():
            return torch.device(f'cuda:{device}')

    def _init_model(self):

        model_type = self.args['model']
        device = self.args['device']
        bn = self.args['bn']

        dropout = self.args['dropout']
        dropout_rate = self.args['dropout_rate']

        hidden_layers = self.args['hidden_layers']
        hidden_units = self.args['hidden_units']

        conv_blocks = self.args['conv_blocks']
        kernel_size = self.args['kernel_size']
        filters = self.args['filters']

        if model_type == 'dense_no_xc':
            return DenseNoXC(device, bn=bn, dropout=dropout, dropout_rate=dropout_rate, hidden_layers=hidden_layers, hidden_units=hidden_units)
        elif model_type == 'dense_ohe':
            return DenseOHE(device, bn=bn, dropout=dropout, dropout_rate=dropout_rate, hidden_layers=hidden_layers, hidden_units=hidden_units)
        elif model_type == 'dense_embed':
            return DenseEmbed(device, bn=bn, dropout=dropout, dropout_rate=dropout_rate, hidden_layers=hidden_layers, hidden_units=hidden_units)
        elif model_type == 'conv1d_embed':
            return Conv1DEmbed(
                device, bn=bn, dropout=dropout, dropout_rate=dropout_rate, hidden_layers=hidden_layers, hidden_units=hidden_units,
                conv_blocks=conv_blocks, kernel_size=kernel_size, filters=filters
            )
        else:
            raise NotImplementedError(
                f'Model type: {model_type} not found. Refer to src.models for models.')

    def _load_model_weights(self):
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def test(self):
        y_pred = []
        scenario_num = []

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for _, (feat_vec, xc, scenario) in enumerate(tqdm(self.test_loader)):
                feat_vec, xc = feat_vec.to(self.device), xc.to(self.device)

                # Make predictionssssss
                outputs = self.model(feat_vec, xc)
                outputs = torch.sigmoid(outputs)
                outputs = torch.round(outputs)

                # Record outputs in list
                preds = outputs.squeeze().cpu().numpy().astype(int)
                scenarios = scenario.squeeze().numpy().astype(int)
                y_pred.extend(preds)
                scenario_num.extend(scenarios)

        result_df = pd.DataFrame(list(zip(scenario_num, y_pred)), columns=[
                                 'dataset_id', 'prediction_score'])
        result_df = result_df.sort_values('dataset_id')
        result_df.to_csv(self.output_path, index=False)
