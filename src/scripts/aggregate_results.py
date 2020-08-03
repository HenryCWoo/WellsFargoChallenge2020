from constants import *

import os
import yaml
from pathlib import Path
import pandas as pd


''' 
! IMPORTANT !
This script must be run from the /src directory

Use: python -m scripts.aggregate_results
'''

all_results = []

for yaml_path in Path(EXPERIMENTS_DIR).glob('**/*.yml'):
    exp_no = yaml_path.parent

    with open(yaml_path, 'r') as yamlfile:
        cur_yaml = yaml.safe_load(yamlfile)

    data = {'exp_path': exp_no}

    for k, v in cur_yaml.items():
        if k == 'cls_report':
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        data[f'{k2} {k3}'] = v3
                else:
                    data[k2] = v2
        else:
            data[k] = v
    all_results.append(data)

all_results = pd.DataFrame(all_results)
all_results = all_results.sort_values(
    by=['model', 'weighted avg f1-score'], ascending=False)

all_results.to_csv(os.path.join(REPORTS_DIR, 'aggregate_experiments.csv'))
