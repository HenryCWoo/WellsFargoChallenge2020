import subprocess
from tqdm import tqdm
from itertools import product

''' 
! IMPORTANT !
This script must be run from the /src directory
'''

models = ['dense_embed']
optims = ['sgd']
epochs = [128]
bns = [True, False]
dropouts = [True, False]
lrs = [1e-2, 1e-3, 1e-4]
hidden_layers = [2, 3, 4]
hidden_units = [16, 64, 128]


flags = ['model', 'optim', 'epochs', 'bn', 'dropout', 'lr', 'hidden_layers',
         'hidden_units']

combinations = [i for i in product(
    models, optims, epochs, bns, dropouts, lrs, hidden_layers, hidden_units)]

print('Total experiments:', len(combinations))

for c in tqdm(combinations):
    cmd = 'python train.py --no_save'

    for i, flag_val in enumerate(c):
        if str(flag_val) == 'True' or str(flag_val) == 'False':
            if str(flag_val) == 'True':
                cmd += f' --{flags[i]}'
        else:
            cmd += f' --{flags[i]}={flag_val}'

    print('\n[COMMAND]', cmd, '\n')
    process = subprocess.call(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
