import subprocess
from tqdm import tqdm
from itertools import product

''' 
! IMPORTANT !
This script must be run from the /src directory
'''

models = ['dense_no_xc', 'dense_ohe', 'dense_embed', 'conv1d_embed']
epochs = [300]
lrs = [1e-2, 1e-3]
step_sizes = [50, 100]
gammas = [0.1, 0.5]

flags = ['model', 'epochs', 'lr', 'step_size', 'gamma']

combinations = [i for i in product(
    models, epochs, lrs, step_sizes, gammas)]

print('Total experiments:', len(combinations))

for c in tqdm(combinations):
    cmd = 'python train.py --scheduler --no_save'

    for i, flag_val in enumerate(c):
        if str(flag_val) == 'True' or str(flag_val) == 'False':
            if str(flag_val) == 'True':
                cmd += f' --{flags[i]}'
        else:
            cmd += f' --{flags[i]}={flag_val}'

    print('\n[COMMAND]', cmd, '\n')
    process = subprocess.call(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
