from constants import *

import os
import argparse
import yaml

from test_model import TestModel
from train_model import TrainModel
from xgboost_trainer import XGBoostTrainer

parser = argparse.ArgumentParser()

# Train or test
parser.add_argument('--test', action='store_true',
                    default=False, help='Use model to predict test data.')

# Model
parser.add_argument('--model', type=str, default='dense_embed',
                    choices=['dense_no_xc', 'dense_ohe',
                             'dense_embed', 'conv1d_embed', 'xgboost'],
                    help='Classifier model.')
parser.add_argument('--hidden_layers', type=int, default=2,
                    help='Hidden layers count to use for Neural Networks.')
parser.add_argument('--hidden_units', type=int, default=64,
                    help='Number of hidden units per layer.')
parser.add_argument('--conv_blocks', type=int, default=1,
                    help='Convolution layer count. (Applies only to convolution networks.)')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='Size of each filter. (Applies only to convolution networks.)')
parser.add_argument('--filters', type=int, default=32,
                    help='Number of filters or channels per convolution block. (Applies only to convolution networks.)')


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
parser.add_argument('--no_save', action='store_true',
                    default=False, help='Choose not to save model parameters.')

# Logging
parser.add_argument('--note', type=str,
                    help='Leave note to describe experiment. (Optional)')

args = parser.parse_args()
args = vars(args)


def save_args():
    ''' Save arguments into yaml file '''
    global args
    with open(os.path.join(EXPERIMENTS_DIR, str(args['exp_no']), 'info.yml'), 'w') as f:
        yaml.dump(args, f, default_flow_style=False)


def load_args():
    ''' Load args if using existing experiment parameters '''
    global args
    with open(os.path.join(EXPERIMENTS_DIR, str(args['exp_no']), 'info.yml'), 'r') as f:
        old_args = yaml.safe_load(f)
        old_args['no_save'] = args['no_save']
        args = old_args
        print('LOADING SAVED PARAMETERS:\n', args)


def increm_experiment_dir():
    ''' Increment and save args['exp_no'] '''
    global args
    exp_nums = []
    for exp_num in os.listdir(EXPERIMENTS_DIR):
        exp_nums.append(int(exp_num))
    next_exp = 0 if len(exp_nums) == 0 else max(exp_nums) + 1
    args['exp_no'] = next_exp


def init_exp_dir():
    ''' Create or load an experiment '''
    global args

    # Create root save model directory if it doesn't exist
    if not os.path.exists(EXPERIMENTS_DIR):
        os.mkdir(EXPERIMENTS_DIR)

    # Create new experiment if exp number was not specified
    if args['exp_no'] == -1:
        increm_experiment_dir()

    exp_dir = os.path.join(EXPERIMENTS_DIR, str(args['exp_no']))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        save_args()
    else:
        load_args()


if __name__ == '__main__':
    if args['test']:
        model = TestModel(args['exp_no'], args['cpu'], args['device'])
        model.test()
    else:
        init_exp_dir()
        if args['model'] == 'xgboost':
            model = XGBoostTrainer(**args)
            model.train()
        else:
            model = TrainModel(**args)
            model.train()
