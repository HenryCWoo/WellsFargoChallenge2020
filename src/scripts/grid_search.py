import subprocess
from tqdm import tqdm
from itertools import product

''' 
! IMPORTANT !
This script must be run from the /src directory
'''

models = ['dense_no_xc', 'dense_ohe', 'dense_embed', 'conv1d_embed']
epochs = [100]
lrs = [1e-2, 1e-3, 1e-4]
bns = [True, False]
dropouts = [True, False]
dropout_rates = [0.5, 0.8]

flags = ['model', 'epochs', 'lr', 'bn', 'dropout', 'dropout_rate']

combinations = [i for i in product(
    models, epochs, lrs, bns, dropouts, dropout_rates)]

print('Total experiments:', len(combinations))

for c in tqdm(combinations):
    cmd = 'python train.py'

    for i, flag_val in enumerate(c):
        if str(flag_val) == 'True' or str(flag_val) == 'False':
            cmd += f' --{flags[i]}'
        else:
            cmd += f' --{flags[i]}={flag_val}'

    print('\n[COMMAND]', cmd, '\n')
    process = subprocess.call(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
