import os
import argparse
import yaml

from train_model import TrainModel

MODEL_DIR = '../models'

parser = argparse.ArgumentParser()

# Model
parser.add_argument('--model', type=str, default='fcn', choices=['fcn'],
                    help='Classifier model.')

# Hyperparameters
parser.add_argument('--epochs', type=int, default=64,
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
parser.add_argument('--loss_fn', type=str, default='x_entropy', choices=['x_entropy'],
                    help='Loss function used for training.')

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
    with open(os.path.join(MODEL_DIR, str(args.exp_no), 'info.yml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def load_args():
    ''' Load arguments from yaml file '''
    yaml_path = os.path.join(MODEL_DIR, str(args.exp_no), 'info.yml')
    with open(yaml_path, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)
        args.__dict__ = data  # TODO: Verify that this works


def increm_experiment_dir():
    ''' Increment and save args.exp_no '''
    exp_nums = []
    for exp_num in os.listdir(MODEL_DIR):
        exp_nums.append(int(exp_num))
    next_exp = 0 if len(exp_nums) == 0 else max(exp_nums) + 1
    args.exp_no = next_exp


def init_exp_dir():
    ''' Create or load an experiment '''
    # Create new experiment if exp number was not specified
    if args.exp_no == -1:
        increm_experiment_dir()

    exp_dir = os.path.join(MODEL_DIR, str(args.exp_no))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        save_args()
    else:
        load_args()


if __name__ == '__main__':
    init_exp_dir()
    model = TrainModel(**vars(args))
    model.train()