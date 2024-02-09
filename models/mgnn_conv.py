from typing import Optional

import torch
import torch.nn.functional as F
from math import log
from torch_geometric.nn import Linear, MLP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch import Tensor

class MGNNConv(MessagePassing):
    def __init__(self, channels, alpha, beta, theta, layer, eps = 1e-5):
        super(MGNNConv, self).__init__(aggr = 'add')
        self.alpha = alpha
        self.beta = beta
        self.gamma = log(theta / layer + 1)
        self.eps = eps
        self.channels = channels

        self.linear = Linear(in_channels = channels, out_channels = channels, bias = True, weight_initializer = 'glorot')
        
        self.reset_parameters()
        
    def reset_parameters(self):
        super().reset_parameters()
        self.linear.reset_parameters()
        
    def forward(self, x, x_0, edge_index, edge_metric, norm, deg_inv_sqrt, edge_mask):
        if isinstance(edge_mask, Tensor):
            self.edge_mask = edge_mask
        else:
            self.register_parameter("edge_mask", None)
        x = self.alpha * x_0 + self.propagate(x = x, edge_index = edge_index, edge_metric = edge_metric, norm = norm, deg_inv_sqrt = deg_inv_sqrt.reshape(-1, 1))
        x = (1 - self.gamma) * x + self.gamma * self.linear.forward(x)
        return x
    
    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None):
        
        if not self.edge_mask is None:
            inputs *= self.edge_mask[:,None]
            
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim)
        
    def message(self, x_i, x_j, edge_metric, norm, deg_inv_sqrt_i, deg_inv_sqrt_j):
        topological_message = norm.view(-1, 1) * x_j
        positional_message = norm.view(-1, 1) * edge_metric.view(-1, 1) * (x_i - x_j) / ( (torch.norm((deg_inv_sqrt_i.view(-1, 1) * x_i - deg_inv_sqrt_j.view(-1, 1) * x_j), p = 2, dim = 1) + self.eps).view(-1, 1) )
        return (1 - self.alpha) * topological_message + self.beta * positional_message

class MGNNAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, method = 'concat'):
        super(MGNNAttention, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.method = method

        assert self.method in ['cosine', 'concat', 'bilinear']

        self.initial = MLP(channel_list = [self.in_channels, self.hidden_channels, self.hidden_channels], dropout = self.dropout, norm = None)
        self.a = Linear(in_channels = 2 * self.hidden_channels, out_channels = 1, bias = False, weight_initializer = 'glorot')
        self.W = Linear(in_channels = self.hidden_channels, out_channels = self.hidden_channels, bias = False, weight_initializer = 'glorot')

    def forward(self, x, edge_index):
        x = self.initial(x)
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        del x

        if self.method == 'cosine':
            edge_attention = F.cosine_similarity(x_i, x_j, dim = 1)
        elif self.method == 'concat':
            edge_attention = torch.tanh(self.a(torch.cat([x_i, x_j], dim = 1))).flatten()
        elif self.method == 'bilinear':
            edge_attention = torch.tanh(torch.sum(x_i * self.W(x_j), dim = 1))
        del x_i, x_j
        return edge_attention