import torch
from torch import nn
from torch.nn import functional as F

from wells_fargo_dataset import XC_COUNT

EMBED_VEC_SIZE = 16


class Conv1DEmbed(nn.Module):
    ''' This model encodes the 'XC' column as an embedding '''

    def __init__(self, device, bn=True, dropout=True, dropout_rate=0.5):
        super(Conv1DEmbed, self).__init__()
        self.device = device
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.bn = bn

        self.embed_xc = nn.Embedding(XC_COUNT, EMBED_VEC_SIZE)
        self.layer_xc = self._embedding_layer(32)

        self.layer_x = self._conv_block(1, 32)
        self.layer_1 = self._conv_block(32, 32)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.dense_1 = self._hidden_layer(32 * 22 + 32, 128)
        self.dense_2 = self._hidden_layer(128, 64)
        self.output = nn.Linear(64, 1)

    def _conv_block(self, input, output):
        modules = []
        modules.append(nn.Conv1d(input, output, kernel_size=5))
        modules.append(nn.ReLU(inplace=True))
        return nn.Sequential(*modules)

    def _hidden_layer(self, input, output):
        modules = []
        modules.append(nn.Linear(input, output))

        if self.bn:
            modules.append(nn.BatchNorm1d(output))

        modules.append(nn.ReLU(inplace=True))

        if self.dropout:
            modules.append(nn.Dropout(self.dropout_rate))

        return nn.Sequential(*modules)

    def _embedding_layer(self, output):
        modules = []
        modules.append(nn.Linear(EMBED_VEC_SIZE, output))

        if self.bn:
            modules.append(nn.BatchNorm1d(output))

        modules.append(nn.ReLU(inplace=True))

        if self.dropout:
            modules.append(nn.Dropout(self.dropout_rate))

        return nn.Sequential(*modules)

    def forward(self, x, xc):
        xc = self.embed_xc(xc.long()).squeeze()
        xc = self.layer_xc(xc)

        x = x.unsqueeze(1)
        x = self.layer_x(x)
        x = self.layer_1(x)

        if self.dropout:
            x = self.dropout(x)
        x = x.view(-1, 32 * 22)

        x = torch.cat((x, xc), dim=1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.output(x)

        return x
