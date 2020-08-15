import torch
from torch import nn
from torch.nn import functional as F


class DenseNoXC(nn.Module):
    ''' This model omits the 'XC' column as input. '''

    def __init__(self, device, bn=True, dropout=True, dropout_rate=0.5, hidden_layers=2, hidden_units=64):
        super(DenseNoXC, self).__init__()
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.bn = bn

        self.hidden_layers_no = hidden_layers
        self.hidden_units_no = hidden_units

        self.layer_1 = self._hidden_layer(30, hidden_units)
        self.hidden_layers = self._hidden_layers()

        self.output = nn.Linear(hidden_units, 1)

    def _hidden_layer(self, input, output):
        modules = []
        modules.append(nn.Linear(input, output))

        if self.bn:
            modules.append(nn.BatchNorm1d(output))

        modules.append(nn.ReLU(inplace=True))

        if self.dropout:
            modules.append(nn.Dropout(self.dropout_rate))

        return nn.Sequential(*modules)

    def _hidden_layers(self):
        modules = []
        for _ in range(self.hidden_layers_no):
            modules.append(self._hidden_layer(
                self.hidden_units_no, self.hidden_units_no))
        return nn.Sequential(*modules)

    def forward(self, x, xc):
        ''' Ignore 'xc' input and only use the non-categorical features '''
        x = self.layer_1(x)
        x = self.hidden_layers(x)
        x = self.output(x)

        return x
