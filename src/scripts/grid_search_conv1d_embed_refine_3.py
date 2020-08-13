import subprocess
from tqdm import tqdm
from itertools import product

''' 
! IMPORTANT !
This script must be run from the /src directory
'''

models = ['conv1d_embed']
optim = ['adam', 'sgd']
epochs = [64, 128]
lrs = [1e-2, 1e-3, 1e-4]
hidden_layers = [2, 3]
hidden_units = [32, 64]
conv_blocks = [1]
kernel_size = [2, 3, 4]
filters = [128]


flags = ['model', 'optim', 'epochs', 'lr', 'hidden_layers',
         'hidden_units', 'conv_blocks', 'kernel_size', 'filters']

combinations = [i for i in product(
    models, optim, epochs, lrs, hidden_layers, hidden_units, conv_blocks, kernel_size, filters)]

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
