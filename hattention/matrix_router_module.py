"""Single-probe, 64-dimensional addressing for matrix LLA-GDN."""
import math
import torch
from torch import nn
from hattention.softmax_matrix_gdn import softmax_matrix_gdn


class MatrixMemoryRouter(nn.Module):
    def __init__(self, hidden_size, heads, key_dim, value_dim):
        super().__init__()
        self.heads, self.value_dim = heads, value_dim
        self.probe = nn.Parameter(torch.empty(heads, key_dim))
        self.query = nn.Linear(hidden_size, heads * value_dim, bias=False)
        self.log_temperature = nn.Parameter(torch.empty(heads))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.probe, std=self.probe.shape[-1] ** -.5)
        nn.init.constant_(self.log_temperature, math.log(self.value_dim ** .5))

    def forward(self, hidden_states, q, k, v, g, beta):
        query = self.query(hidden_states).reshape(*hidden_states.shape[:2], self.heads, self.value_dim)
        return softmax_matrix_gdn(q,k,v,g,beta,query,self.probe,self.log_temperature)
