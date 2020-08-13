import os
import yaml
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from constants import *

from wells_fargo_dataset import WellsFargoDataset, CHAR_COL

TRAIN_DATA = '../data/raw/0173eeb640e7-Challenge+Data+Set+-+Campus+Analytics+2020.xlsx'

# Set Seeds so that experiments are reproduceable
np.random.seed(seed=123)


class XGBoostTrainer:
    def __init__(self, **args):
        # Save arguments for reinitializing model during k-fold validation
        self.args = args

        self.exp_no = args['exp_no']
        self.info_yml = os.path.join(
            EXPERIMENTS_DIR, str(self.exp_no), 'info.yml')

        # Dataset
        self.data_df = self._init_dataframe(TRAIN_DATA)

        self.kfold = args['kfold']

    def _init_dataframe(self, excel_path):
        df = pd.read_excel(excel_path)

        cat_type = CategoricalDtype(
            categories=['A', 'B', 'C', 'D', 'E'], ordered=True)
        df[CHAR_COL] = df[CHAR_COL].astype(cat_type).cat.codes
        return df

    def _get_classification_report(self, model, x_valid, y_valid, output_dict=True):
        predictions = model.predict(x_valid)
        predictions = [round(value) for value in predictions]
        return classification_report(y_valid, predictions, zero_division=0, output_dict=output_dict)

    def _kfold_dataset_generator(self, shuffle_dataset=True):
        # Creating data indices for training and validation splits:
        dataset_size = len(self.data_df.index)
        print(dataset_size)
        indices = list(range(dataset_size))
        split = int(np.floor(dataset_size / float(self.kfold)))
        if shuffle_dataset:
            np.random.shuffle(indices)
        for fold_no in range(self.kfold):
            train_indices = indices[: fold_no * split] + \
                indices[min((fold_no + 1) * split, dataset_size):]
            test_indices = indices[fold_no * split:
                                   min((fold_no + 1) * split, dataset_size)]

            x_train = self.data_df.iloc[train_indices].drop(['y'], axis=1)
            y_train = self.data_df.iloc[train_indices].loc[:, 'y']

            x_valid = self.data_df.iloc[test_indices].drop(['y'], axis=1)
            y_valid = self.data_df.iloc[test_indices].loc[:, 'y']

            yield x_train, y_train, x_valid, y_valid

    def _average_cls_reports(self, cls_reports):
        if not cls_reports:
            return None

        if len(cls_reports) == 1:
            return cls_reports[0]

        result = cls_reports[0]
        for cls_report in cls_reports[1:]:
            for k, v in cls_report.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        result[k][k2] += v2
                else:
                    result[k] += v

        for k, v in result.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    result[k][k2] /= len(cls_reports)
            else:
                result[k] /= len(cls_reports)

        return result

    def train(self):
        print('k-Fold Cross Validation start.')

        # Collect classification reports of each k-Fold
        cls_reports = []
        for fold_no, (x_train, y_train, x_valid, y_valid) in enumerate(self._kfold_dataset_generator()):
            # print(len(x_train.index))
            counts = y_train.value_counts()
            counts_0, counts_1 = counts[0], counts[1]
            imbalance_ratio = counts_0 / counts_1
            model = XGBClassifier(scale_pos_weight=imbalance_ratio)
            model.fit(x_train, y_train)

            print(
                f'\n[Classification Report for K-fold {fold_no + 1}]\n',
                self._get_classification_report(
                    model, x_valid, y_valid, output_dict=False),
                '\n',
            )

            # Save Classification Report in info.yml
            cls_report = self._get_classification_report(
                model, x_valid, y_valid, output_dict=True)
            cls_reports.append(cls_report)

        avg_cls_report = self._average_cls_reports(cls_reports)
        with open(self.info_yml, 'r+') as f:
            info_yml = yaml.safe_load(f)
            info_yml.update(
                {'cls_report': avg_cls_report})
            yaml.safe_dump(info_yml, f)

        print('Finished k-Fold Cross Validation.')

        # Build Final Model
        train_data = self.data_df.drop(['y'], axis=1)
        y_counts = self.data_df.loc[:, 'y']
        imbalance_ratio = y_counts[0] / y_counts[1]

        model = XGBClassifier(scale_pos_weight=imbalance_ratio)
        model.fit(train_data, y_counts)

        print(
            f"AVERAGE WEIGHTED F-1 SCORE: {avg_cls_report['weighted avg']['f1-score']}")
