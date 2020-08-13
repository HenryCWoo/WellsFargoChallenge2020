import subprocess
from tqdm import tqdm
from itertools import product

''' 
! IMPORTANT !
This script must be run from the /src directory
'''

models = ['conv1d_embed']
epochs = [128, 256]
lrs = [1e-3, 1e-4]
bns = [True, False]
dropout = [True, False]
dropout_rates = [0.5, 0.8]
hidden_layers = [2, 3, 4]
hidden_units = [64, 128]
conv_blocks = [1]
kernel_size = [2, 3, 4]
filters = [128]


flags = ['model', 'epochs', 'lr', 'bn', 'dropout', 'dropout_rate', 'hidden_layers',
         'hidden_units', 'conv_blocks', 'kernel_size', 'filters']

combinations = [i for i in product(
    models, epochs, lrs, bns, dropout, dropout_rates, hidden_layers, hidden_units, conv_blocks, kernel_size, filters)]

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
