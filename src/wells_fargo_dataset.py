import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

TARGET_COL = 'y'
CHAR_COL = 'XC'


class WellsFargoDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.transform = transform

        # Assume target / label column is labelled 'y'
        self.targets = data_df[TARGET_COL].to_numpy()

        # Extract feature column containing chars
        # User may want to use embeddings or one-hot encoding, etc.
        self.chars = data_df[CHAR_COL].to_numpy()

        # Remove label column and get features (excluding the char column)
        feat_df = data_df.drop([TARGET_COL, CHAR_COL], axis=1)
        self.feat_vecs = feat_df.to_numpy()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        feat_vec, char, target = self.feat_vecs[idx], self.chars[idx], self.targets[idx]
        feat_vec, target = torch.Tensor(feat_vec), torch.Tensor([target])

        if self.transform is not None:
            feat_vec = self.transform(feat_vec)

        return feat_vec, char, target


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_df = pd.read_excel(
        './data/raw/0173eeb640e7-Challenge+Data+Set+-+Campus+Analytics+2020.xlsx')
    ds = WellsFargoDataset(data_df)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=5, shuffle=False, num_workers=0)

    for batch_idx, (vec, char, target) in enumerate(dl):
        print(batch_idx, vec, char, target)
