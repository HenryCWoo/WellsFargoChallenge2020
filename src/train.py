from constants import *

import os
import argparse
import yaml

from train_model import TrainModel
from xgboost_trainer import XGBoostTrainer

parser = argparse.ArgumentParser()

# Model
parser.add_argument('--model', type=str, default='dense_embed',
                    choices=['dense_no_xc', 'dense_ohe',
                             'dense_embed', 'conv1d_embed', 'xgboost'],
                    help='Classifier model.')

# Hyperparameters
parser.add_argument('--kfold', type=int, default=5,
                    help='Number of folds for k-Fold cross validation.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of total epochs in classifier training.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Classifier learning rate.')
parser.add_argument('--optim', type=str, default='adam',
                    choices=['sgd', 'adam'], help='Optimizer method.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--bn', action='store_true', default=False,
                    help='Use batch normalization in models.')
parser.add_argument('--dropout', action='store_true',
                    default=False, help='Use dropout layers.')
parser.add_argument('--dropout_rate', type=float, default=0.5,
                    help='Dropout rate for the layers.')
parser.add_argument('--loss_fn', type=str, default='bce', choices=['bce'],
                    help='Loss function used for training.')
parser.add_argument('--scheduler', action='store_true',
                    default=False, help='Use learning rate scheduler.')
parser.add_argument('--step_size', type=int, default=32,
                    help='(--scheduler flag must be set) Interval of epochs until learning rate change.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='(--scheduler flag must be set) Change in learning rate factor.')

# Others
parser.add_argument('--exp_no', type=int, default=-1,
                    help='Experiment number.')
parser.add_argument('--device', type=int, default=0,
                    help='GPU device number if using GPU resources.')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='Train using CPU instead of GPU resources.')

# Logging
parser.add_argument('--note', type=str,
                    help='Leave note to describe experiment. (Optional)')

args = parser.parse_args()


def save_args():
    ''' Save arguments into yaml file '''
    with open(os.path.join(EXPERIMENTS_DIR, str(args.exp_no), 'info.yml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def load_args():
    ''' Load arguments from yaml file '''
    yaml_path = os.path.join(EXPERIMENTS_DIR, str(args.exp_no), 'info.yml')
    with open(yaml_path, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)
        args.__dict__ = data  # TODO: Verify that this works


def increm_experiment_dir():
    ''' Increment and save args.exp_no '''
    exp_nums = []
    for exp_num in os.listdir(EXPERIMENTS_DIR):
        exp_nums.append(int(exp_num))
    next_exp = 0 if len(exp_nums) == 0 else max(exp_nums) + 1
    args.exp_no = next_exp


def init_exp_dir():
    ''' Create or load an experiment '''
    # Create new experiment if exp number was not specified
    # Create root save model directory if it doesn't exist
    if not os.path.exists(EXPERIMENTS_DIR):
        os.mkdir(EXPERIMENTS_DIR)

    if args.exp_no == -1:
        increm_experiment_dir()

    exp_dir = os.path.join(EXPERIMENTS_DIR, str(args.exp_no))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        save_args()
    else:
        load_args()


if __name__ == '__main__':
    # init_exp_dir()
    if args.model == 'xgboost':
        model = XGBoostTrainer(**vars(args))
        model.train()
    else:
        model = TrainModel(**vars(args))
        model.train()
