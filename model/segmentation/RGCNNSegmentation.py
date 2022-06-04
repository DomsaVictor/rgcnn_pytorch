import sys
from pathlib import Path
cwd = Path.cwd()
curr_path = Path(__file__).parent
utils_path = (curr_path / "../../utils/").resolve()
sys.path.append(str(utils_path))
import utils as conv

import torch
from torch import nn
from torch.nn.functional import one_hot, relu

class seg_model(nn.Module):
    def __init__(self, vertice, F, K, M, input_dim=22, one_layer=False, dropout=1, reg_prior: bool = True, b2relu=True, recompute_L=False, fc_bias=True):
        assert len(F) == len(K)
        super(seg_model, self).__init__()

        self.F = F
        self.K = K
        self.M = M
        self.one_layer = one_layer
        self.reg_prior = reg_prior
        self.vertice = vertice
        self.dropout = dropout
        self.recompute_L = recompute_L
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.relus = self.F + self.M

        if b2relu:
            self.bias_relus = nn.ParameterList([
                torch.nn.parameter.Parameter(torch.zeros((1, vertice, i))) for i in self.relus
            ])
        else:
            self.bias_relus = nn.ParameterList([
                torch.nn.parameter.Parameter(torch.zeros((1, 1, i))) for i in self.relus
            ])

        self.conv = nn.ModuleList([
            conv.DenseChebConvV2(input_dim, self.F[i], self.K[i]) if i == 0 else conv.DenseChebConvV2(self.F[i-1], self.F[i], self.K[i]) for i in range(len(K))
        ])

        self.fc = nn.ModuleList([])
        for i in range(len(M)):
            if i == 0:
                self.fc.append(nn.Linear(self.F[-1], self.M[i], fc_bias))
            elif i == 1:
                self.fc.append(
                    nn.Linear(self.M[i-1]+self.M[i-1], self.M[i], fc_bias))
            else:
                self.fc.append(nn.Linear(self.M[i-1], self.M[i], fc_bias))

        self.L = []
        self.x = []

    def b1relu(self, x, bias):
        return relu(x + bias)

    def brelu(self, x, bias):
        return relu(x + bias)

    def get_laplacian(self, x):
        with torch.no_grad():
            return conv.get_laplacian(conv.pairwise_distance(x))

    @torch.no_grad()
    def append_regularization_terms(self, x, L):
        if self.reg_prior:
            self.L.append(L)
            self.x.append(x)

    @torch.no_grad()
    def reset_regularization_terms(self):
        self.L = []
        self.x = []

    def forward(self, x, cat=None):
        self.reset_regularization_terms()

        x1 = 0  # cache for layer 1

        L = self.get_laplacian(x)

        if cat is not None:
            cat = one_hot(cat, num_classes=16)
            cat = torch.tile(cat, [1, self.vertice, 1])
            out = torch.cat([x, cat], dim=2)  # Pass this to the model
        else:
            out = x

        for i in range(len(self.K)):
            out = self.conv[i](out, L)
            self.append_regularization_terms(out, L)
            if self.recompute_L:
                L = self.get_laplacian(out)
            out = self.dropout(out)
            out = self.brelu(out, self.bias_relus[i])
            if i == 1:
                x1 = out

        for i in range(len(self.M)):
            if i == 1:
                out = torch.concat([out, x1], dim=2)
            out = self.fc[i](out)
            self.append_regularization_terms(out, L)
            out = self.dropout(out)
            out = self.b1relu(out, self.bias_relus[i + len(self.K)])

        return out, self.x, self.L