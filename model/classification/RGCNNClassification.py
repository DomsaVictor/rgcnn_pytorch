import sys
from pathlib import Path
cwd = Path.cwd()
curr_path = Path(__file__).parent
utils_path = (curr_path / "../../utils/").resolve()
sys.path.append(str(utils_path))
import utils as conv

import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn.functional import one_hot, relu, leaky_relu

class cls_model(nn.Module):
    def __init__(self, vertice, F, K, M, input_dim=22, one_layer=False, dropout=1, reg_prior: bool = True, b2relu=True, recompute_L=False, fc_bias=True, cheb_bias=True):
        assert len(F) == len(K)
        super(cls_model, self).__init__()

        self.F = F
        self.K = K
        self.M = M
        self.one_layer = one_layer
        self.reg_prior = reg_prior
        self.vertice = vertice
        self.dropout = dropout
        self.recompute_L = recompute_L
        self.dropout = torch.nn.Dropout(p=self.dropout)
        # self.dropout = torch.nn.Dropout2d(p=self.dropout)
        self.relus = self.F + self.M
        self.cheb_bias = cheb_bias
        
        if b2relu:
            self.bias_relus = nn.ParameterList([
                torch.nn.parameter.Parameter(torch.zeros((1, vertice, i))) for i in self.F])
        else:
            self.bias_relus = nn.ParameterList([
                torch.nn.parameter.Parameter(torch.zeros((1, 1, i))) for i in self.F])

        for i in self.M:
            self.bias_relus.append(torch.nn.parameter.Parameter(torch.zeros((1, i))))

        self.conv = nn.ModuleList([
            conv.DenseChebConvV2(input_dim, self.F[i], self.K[i], bias=self.cheb_bias) if i == 0 else conv.DenseChebConvV2(self.F[i-1], self.F[i], self.K[i], bias=self.cheb_bias) for i in range(len(K))])

        self.batch_norm_list_conv = nn.ModuleList([BatchNorm1d(input_dim)])

        for i in range(len(F)):
            self.batch_norm_list_conv.append(nn.BatchNorm1d(F[i])) 

        self.batch_norm_list_fc = nn.ModuleList([
            BatchNorm1d(M[i]) for i in range(len(M))])

        self.fc = nn.ModuleList([])
        for i in range(len(M)):
            if i == 0:
                self.fc.append(nn.Linear(self.F[-1], self.M[i], fc_bias))
            else:
                self.fc.append(nn.Linear(self.M[i-1], self.M[i], fc_bias))

        self.L = []
        self.x = []

    def b1relu(self, x, bias):
        return relu(x + bias)

    def brelu(self, x, bias):
        return leaky_relu(x + bias)

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

    def forward(self, x):
        self.reset_regularization_terms()
        out = x
        # out = self.batch_norm_list_conv[0](out.transpose(1, 2))
        # out = out.transpose(2, 1)
        # L = self.get_laplacian(x)
        L = self.get_laplacian(out[:,:,:3])
        
        for i in range(len(self.K)):
            out = self.conv[i](out, L)
            self.append_regularization_terms(out, L)
            if self.recompute_L:
                L = self.get_laplacian(out)
            # out = self.dropout(out)
            # out = self.b1relu(out, self.bias_relus[i])
            out = relu(out)

            # out = self.batch_norm_list_conv[i+1](out.transpose(1, 2))
            # out = out.transpose(1, 2)

        out, _ = torch.max(out, 1)
        
        for i in range(len(self.M) -1):
            out = self.fc[i](out)
            # self.append_regularization_terms(out, L)
            out = self.dropout(out)
            # out = self.brelu(out, self.bias_relus[i + len(self.K)])
            out = relu(out)
            # out = self.batch_norm_list_fc[i](out)
            # out = out.transpose(0, 1)

        out = self.fc[-1](out)

        return out, self.x, self.L
    
if __name__ == "__main__":
    
    modelnet_num = 40
    
    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, modelnet_num]

    model = cls_model(512, F, K, M, 6, dropout=0.2)
    
    print(model)