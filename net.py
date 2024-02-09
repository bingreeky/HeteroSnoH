import torch
import torch.nn as nn
import pdb
import copy
import math
import utils
import scipy.optimize as optimize
# from torch_geometric.nn import GCNConv
from models.basic_gnn import GAT
from layers import BinaryStep, MaskedLinear
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_geometric.nn as geo_nn
from torch_geometric.nn import MLP, Linear, JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj, to_undirected, degree
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
import torch_sparse

from models.gat_conv import GATConv
from models.gcn2_conv import GCN2Conv
from models.mixhop_conv import MixHopConv
from models.gcn_conv import GCNConv
from models.gin_conv import GINConv
from models.point_conv import PointNetConv
from models.pgnn_conv import pGNNConv
from models.mgnn_conv import MGNNConv, MGNNAttention
from models.sg_conv import SGConv

import matplotlib.pyplot as plt


def net_selector(net_name:str):
    if net_name in ["gcn", "resgcn"]:
        # return net_gcn_dense
        return net_gcn
    elif net_name in ['jknet']:
        return net_jknet
    elif net_name in ['gat']:
        return net_gat
    elif net_name in ['gcn2', 'gcnii']:
        return net_gcn2
    elif net_name in ['mixhop']:
        return net_mixhop
    elif net_name in ['gin']:
        return net_gin
    elif net_name in ["gpnn"]:
        return net_point
    elif net_name in ['pgnn']:
        return net_pgnn
    elif net_name in ['mgnn']:
        return net_mgnn
    elif net_name in ['h2gcn']:
        return net_h2gcn
    elif net_name in ['sgc']:
        return net_sgc
    elif net_name in ['appnp']:
        return net_appnp


class net_gcn_dense(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, coef=None, baseline=False, mode="prune"):
        super().__init__()

        self.mode = mode
        self.adj_binary = to_dense_adj(edge_index)[0].to(device)
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.coef = coef
        
        self.use_bn = use_bn
        self.use_res = use_res
        self.baseline = baseline
        out_channel = embedding_dim[-1]
        
        if self.use_bn:
            self.norms = nn.ModuleList()
            for i in range(self.layer_num):
                self.norms.append(nn.BatchNorm1d(embedding_dim[i]))

        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.normalize = utils.torch_normalize_adj
        self.device = device
        
        self.scaling_factor = 2
        self.pseudo_mlp = nn.Sequential(
            nn.Linear(embedding_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_channel)
        )
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_channel, out_channel)) / self.scaling_factor) for i in range(self.layer_num)
        ])
    
    def __create_learner_input(self, edge_index, embeds):
        row, col = edge_index[0], edge_index[1]
        row_embs, col_embs = embeds[row], embeds[col]
        edge_learner_input = torch.cat([row_embs, col_embs], 1)
        return edge_learner_input
    
    def prepare_masked_adj(self, edge_masks):
        self.adj_ls = []
        for i in range(self.layer_num):
            self.adj_ls.append(self.adj_binary * edge_masks[i])
            self.adj_ls[-1].fill_diagonal_(1)
            self.adj_ls[-1] = self.normalize(self.adj_ls[-1], self.device)

    def forward_retain(self, x, edge_index, val_test, edge_masks):
        
        if not hasattr(self, "adj_ls"):
            self.prepare_masked_adj(edge_masks)
        
        # adj_ori = self.adj_binary
        for ln in range(self.layer_num):
            # adj = adj_ori * edge_masks[ln]
            # # print((adj != adj_ori).sum())
            # adj.fill_diagonal_(1)
            # adj = self.normalize(adj, device=self.device)
            if ln and self.use_bn: x = self.norms[ln](x)
            x = torch.mm(self.adj_ls[ln], x)
            x = self.net_layer[ln](x)
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if not val_test:
                x = self.dropout(x)
            if ln and self.use_res: x += h
        return x

    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
        # return self.forward_retain(x, edge_index, val_test, None)
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])

        edge_weight = None
        if not self.baseline:
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
        self.edge_archive = []
        
        adj_ori = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
        adj = adj_ori
        # adj = self.normalize(adj, self.device) # if not kwargs['pretrain'] else adj_ori
        # adj_ori = self.adj_binary 
        for ln in range(self.layer_num):
            # self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index, ln)
            # self.edge_archive.append(edge_weight.detach().cpu().unsqueeze(0))
            # adj_ori = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
            
            if ln and self.use_bn: x = self.norms[ln](x)
            x = torch.mm(adj, x)
            x = self.net_layer[ln](x)
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if not val_test:
                x = self.dropout(x)
            if ln and self.use_res: x += h
        return x
    
    def learn_soft_edge3(self, x, edge_index, ln=0):
        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        # temp softmax
        # deg = torch.zeros(edge_index.max().item() + 1,
        #                   dtype=torch.float, device=self.device)
        # exp_wei = torch.exp(edge_weight / 3) # Citeseer 1.5 Cora 3
        # deg.scatter_add_(0, edge_index[0], exp_wei)
        # edge_weight = exp_wei / (deg[edge_index[0]] )  # 计算每个边的权重
        return edge_weight
        
    
    def learn_soft_edge(self, x, edge_index, ln=0):
        input = self.__create_learner_input(edge_index, x)
        if ln == 0:
            edge_weight =  self.edge_learner(input).squeeze(-1)
        elif ln == 1:
            edge_weight =  self.edge_learner2(input).squeeze(-1)
        else:
            raise NotImplementedError
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        return edge_weight
    
    def learn_soft_edge2(self, x, edge_index):
        row, col = edge_index
        # print(x.device)
        row_embs, col_embs = self.sim_learner_src(x[row]), self.sim_learner_tgt(x[col])
        # left = torch.einsum("ik,kk->ik",row_embs,self.mid_learner)
        edge_weight =  torch.einsum("ik,ik->i",row_embs, col_embs)
        deg = torch.zeros(edge_index.max().item() + 1,
                          dtype=torch.float, device=self.device)

        exp_wei = torch.exp(edge_weight / 3)
        deg.scatter_add_(0, edge_index[0], exp_wei)
        edge_weight = exp_wei / (deg[edge_index[0]] ) 

        return edge_weight


    def adj_pruning(self, adj, thres, prev_mask):
        mask = BinaryStep.apply(adj - utils.log_custom(thres).view(-1,1))
        return mask * prev_mask if prev_mask is not None else mask
    
    def adj_pruning2(self, adj, thres, prev_mask, tau=0.1, val_test=False):
        edge_weight = adj[adj.nonzero(as_tuple=True)]
        edge_index = adj.nonzero().t().contiguous()
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.01 / variance))
        adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
        B = adj.size(0)
        thres_trans = lambda x: self.coef*(torch.pow(x,3) + 20*x)     
        y_soft = torch.sigmoid((adj - thres_trans(thres)) / tau)
        # y_hrad = (y_soft > 0.5).float()
        y_hrad = ((y_soft + torch.eye(adj.shape[0]).to(self.device))  > 0.5).float()
        ret = y_hrad - y_soft.detach() + y_soft
        return ret * prev_mask # if prev_mask is not None else ret



class net_gcn(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, baseline=False, mode="prune"):
        super().__init__()

        self.mode = mode
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.num_nodes = num_nodes
        self.use_bn = use_bn
        self.use_res = True # TODO
        self.baseline = baseline
        self.relu = nn.ReLU(inplace=True)
        
        if self.use_bn:
            self.norms = nn.ModuleList()
            for i in range(self.layer_num):
                self.norms.append(nn.BatchNorm1d(embedding_dim[i+1]))
            # print(self.norms)

        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        dropout = dropout
        self.graph_norm = False
        self.batch_norm = False
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.layers = nn.ModuleList()

        for ln in range(self.layer_num):
            if ln == 0:
                self.layers.append(GCNConv(in_dim_node, hidden_dim))
            elif ln == self.layer_num - 1:
                self.layers.append(GCNConv(hidden_dim, out_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.normalize = utils.torch_normalize_adj
        self.device = device
        
        self.scaling_factor = 2
        self.pseudo_mlp = nn.Sequential(
            nn.Linear(embedding_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_dim, out_dim)) / self.scaling_factor) for i in range(self.layer_num)
        ])
    

    def forward_retain(self, x, edge_index, val_test, edge_masks):
        
        for ln in range(self.layer_num):
            
            x = self.layers[ln](x, edge_index , edge_mask=edge_masks[ln])
            if ln and self.use_bn: x = self.norms[ln](x)
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if ln and self.use_res: x += h
        return x

    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])

        edge_weight = None
        if (not self.baseline) and (not self.mode == "retain"):
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
        self.edge_archive = []
        
        for ln in range(self.layer_num):
            
            x = self.layers[ln](x, edge_index, edge_mask=edge_weight)
            if ln and self.use_bn: x = self.norms[ln](x)
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if ln and self.use_res: x += h
        return x
    
    def learn_soft_edge3(self, x, edge_index, ln=0):
        
        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        return edge_weight



class net_jknet(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, baseline=False, mode="prune", jk="cat"):
        super().__init__()

        self.mode = mode
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.num_nodes = num_nodes
        self.use_bn = use_bn
        self.use_res = use_res
        self.baseline = baseline
        self.relu = nn.ReLU(inplace=True)
        
        if self.use_bn:
            self.norms = nn.ModuleList()
            for i in range(self.layer_num):
                if i == self.layer_num - 1:
                    self.norms.append(nn.BatchNorm1d(embedding_dim[1]))
                else:
                    self.norms.append(nn.BatchNorm1d(embedding_dim[i+1]))
            # print(self.norms)

        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        dropout = dropout
        self.graph_norm = False
        self.batch_norm = False
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.layers = nn.ModuleList()

        for ln in range(self.layer_num):
            if ln == 0:
                self.layers.append(GCNConv(in_dim_node, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.jump = JumpingKnowledge(jk)
        if jk == "cat":
            self.lin1 = nn.Linear((self.layer_num-1)*hidden_dim, hidden_dim)
        else:
            self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

        self.normalize = utils.torch_normalize_adj
        self.device = device
        
        self.scaling_factor = 2
        self.pseudo_mlp = nn.Sequential(
            nn.Linear(embedding_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_dim, out_dim)) / self.scaling_factor) for i in range(self.layer_num)
        ])
    

    def forward_retain(self, x, edge_index, val_test, edge_masks):
        x = F.relu(self.layers[0](x, edge_index, edge_mask=edge_masks[0]))
        xs = [x]
        for ln in range(1, self.layer_num):
            
            x = self.layers[ln](x, edge_index, edge_mask=edge_masks[ln])
            if ln and self.use_bn: x = self.norms[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            xs += [x]
        x = self.jump(xs)
        x = self.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])

        edge_weight = None
        if (not self.baseline) and (not self.mode == "retain"):
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
        self.edge_archive = []
        
        x = F.relu(self.layers[0](x, edge_index, edge_mask=edge_weight))
        xs = [x]
        for ln in range(1, self.layer_num):
            
            x = self.layers[ln](x, edge_index, edge_mask=edge_weight)
            if ln and self.use_bn: x = self.norms[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            xs += [x]
        x = self.jump(xs)
        x = self.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x
    
    def learn_soft_edge3(self, x, edge_index, ln=0):
        
        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        return edge_weight



class net_gin(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, baseline=False, mode="prune"):
        super().__init__()

        self.mode = mode
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.num_nodes = num_nodes
        self.use_bn = use_bn
        self.use_res = use_res
        self.baseline = baseline
        self.relu = nn.ReLU(inplace=True)
        
        if self.use_bn:
            self.norms = nn.ModuleList()
            for i in range(self.layer_num):
                self.norms.append(nn.BatchNorm1d(embedding_dim[1]))
            # print(self.norms)

        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        dropout = dropout
        self.graph_norm = False
        self.batch_norm = False
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.layers = nn.ModuleList()
        
        for i in range(self.layer_num):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(in_dim_node, 2 * hidden_dim),
                    nn.BatchNorm1d(2 * hidden_dim),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.BatchNorm1d(2 * hidden_dim),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                )
            self.layers.append(GINConv(mlp, train_eps=True))# .jittable()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

        self.normalize = utils.torch_normalize_adj
        self.device = device
        
        self.scaling_factor = 2
        self.pseudo_mlp = nn.Sequential(
            nn.Linear(embedding_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_dim, out_dim)) / self.scaling_factor) for i in range(self.layer_num)
        ])
    

    def forward_retain(self, x, edge_index, val_test, edge_masks):
        
        for ln in range(self.layer_num):
            
            x = self.layers[ln](x, edge_index , edge_mask=edge_masks[ln])
            if ln and self.use_bn: x = self.norms[ln](x)
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if ln and self.use_res: x += h
        return x

    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])

        edge_weight = None
        if (not self.baseline) and (not self.mode == "retain"):
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
        self.edge_archive = []
        
        for ln in range(self.layer_num):
            
            x = self.layers[ln](x, edge_index, edge_mask=edge_weight)
            if ln and self.use_bn: x = self.norms[ln](x)
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if ln and self.use_res: x += h
        return x
    
    def learn_soft_edge3(self, x, edge_index, ln=0):
        
        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        return edge_weight



class net_gat(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, baseline=False, mode="prune"):
        super().__init__()

        self.mode = mode
        # self.adj_binary = to_dense_adj(edge_index)[0].to(device)
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.num_nodes = num_nodes
        self.use_bn = use_bn
        self.use_res = use_res
        self.baseline = baseline
        self.relu = nn.ReLU(inplace=True)
        
        if self.use_bn:
            self.norms = nn.ModuleList()
            for i in range(self.layer_num):
                self.norms.append(nn.BatchNorm1d(embedding_dim[i]))

        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        num_heads = 1
        dropout = dropout
        self.graph_norm = False
        self.batch_norm = False
        self.residual = True
        self.dropout = nn.Dropout(p=dropout)
        self.n_classes = n_classes
        
        self.layers = nn.ModuleList()

        for ln in range(self.layer_num):
            if ln == 0:
                self.layers.append(GATConv(in_dim_node, hidden_dim, heads=num_heads, dropout=dropout))
            elif ln == self.layer_num - 1:
                self.layers.append(GATConv(hidden_dim * num_heads, out_dim, heads=1, dropout=0))
            else:
                self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
                    
        self.normalize = utils.torch_normalize_adj
        self.device = device
        
        self.scaling_factor = 2
        self.pseudo_mlp = nn.Sequential(
            nn.Linear(embedding_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_dim, out_dim)) / self.scaling_factor) for i in range(self.layer_num)
        ])
    
    def prepare_masked_adj(self, edge_masks):
        self.adj_ls = []
        for i in range(self.layer_num):
            self.adj_ls.append(self.adj_binary * edge_masks[i])
            self.adj_ls[-1].fill_diagonal_(1)
            self.adj_ls[-1] = self.normalize(self.adj_ls[-1], self.device)

    def forward_retain(self, x, edge_index, val_test, edge_masks):
        
        # if not hasattr(self, "adj_ls"):
        #     self.prepare_masked_adj(edge_masks)
        
        for ln in range(self.layer_num):
            
            if ln and self.use_bn: x = self.norms[ln](x)
            x = self.layers[ln](x, edge_index, edge_mask=edge_masks[ln])
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            # if not val_test:
            #     x = self.dropout(x)
            if ln and self.use_res: x += h
        return x

    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])

        edge_weight = None
        if not self.baseline:
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
        self.edge_archive = []
        
        for ln in range(self.layer_num):
            
            if ln and self.use_bn: x = self.norms[ln](x)
            x = self.layers[ln](x, edge_index, edge_mask=edge_weight)
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            # if not val_test:
            #     x = self.dropout(x)
            if ln and self.use_res: x += h
        return x
    
    def learn_soft_edge3(self, x, edge_index, ln=0):
        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index,  fill_value="mean",
        #             num_nodes=self.num_nodes)
        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        # temp softmax
        # deg = torch.zeros(edge_index.max().item() + 1,
        #                   dtype=torch.float, device=self.device)
        # exp_wei = torch.exp(edge_weight / 3) # Citeseer 1.5 Cora 3
        # deg.scatter_add_(0, edge_index[0], exp_wei)
        # edge_weight = exp_wei / (deg[edge_index[0]] )  # 计算每个边的权重
        return edge_weight



class net_gcn2(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, baseline=False, mode="prune"):
        super().__init__()

        self.mode = mode
        # self.adj_binary = to_dense_adj(edge_index)[0].to(device)
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.num_nodes = num_nodes
        self.use_bn = use_bn
        self.use_res = use_res
        self.baseline = baseline
        self.relu = nn.ReLU(inplace=True)
        
        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        dropout = dropout
        self.graph_norm = False
        self.batch_norm = False
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim_node, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, n_classes))
        
        if self.use_bn:
            self.norms = nn.ModuleList()
            for i in range(self.layer_num):
                self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.layers = nn.ModuleList()

        for ln in range(self.layer_num):
            self.layers.append(GCN2Conv(hidden_dim, alpha=0, theta=1.0, layer=ln+1, shared_weights=False))
                    
        self.normalize = utils.torch_normalize_adj
        self.device = device
        
        self.scaling_factor = 2
        self.pseudo_mlp = nn.Sequential(
            nn.Linear(embedding_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_dim, out_dim)) / self.scaling_factor) for i in range(self.layer_num)
        ])

    def forward_retain(self, x, edge_index, val_test, edge_masks):
        
        x = x_0 = self.lins[0](x).relu()
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.norms[1](x)
        for idx, conv in enumerate(self.layers):
            x = conv(x, x_0, edge_index, edge_mask=edge_masks[idx]) # edge_masks[idx]
            # h = self.norms[idx](h)
            x = F.dropout(x, self.dropout, training=self.training)
            x = x.relu()
            
        x = self.lins[1](x)
        return x
        
    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])

        edge_weight = None
        if not self.baseline:
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
            # print(self.)
        self.edge_archive = []

        x = x_0 = self.lins[0](x).relu()
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.norms[1](x)
        for idx, conv in enumerate(self.layers):
            x = conv(x, x_0, edge_index, edge_mask=edge_weight) # edge_masks[idx]
            # h = self.norms[idx](h)
            x = F.dropout(x, self.dropout, training=self.training)
            x = x.relu()
            
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        
        return x
        
    
    def learn_soft_edge3(self, x, edge_index, ln=0):
        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index,  fill_value="mean",
        #             num_nodes=self.num_nodes)
        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        # temp softmax
        # deg = torch.zeros(edge_index.max().item() + 1,
        #                   dtype=torch.float, device=self.device)
        # exp_wei = torch.exp(edge_weight / 3) # Citeseer 1.5 Cora 3
        # deg.scatter_add_(0, edge_index[0], exp_wei)
        # edge_weight = exp_wei / (deg[edge_index[0]] )  # 计算每个边的权重
        return edge_weight


"""
https://github.com/syleeheal/AERO-GNN/blob/609423b3117d2b285c787dfc5b0e29917eb447a1/AERO-GNN/models.py
"""
class net_mixhop(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, baseline=False, mode="prune"):
        super().__init__()

        self.mode = mode
        # self.adj_binary = to_dense_adj(edge_index)[0].to(device)
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.num_nodes = num_nodes
        self.use_bn = use_bn
        self.use_res = use_res
        self.baseline = baseline
        self.relu = nn.ReLU(inplace=True)
        
        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        dropout = dropout
        self.graph_norm = False
        self.batch_norm = False
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        
        
        self.norms = nn.ModuleList()
        # if self.use_bn:
        #     for i in range(self.layer_num):
        #         self.norms.append(nn.BatchNorm1d(embedding_dim[i]))

        hops=[0,1,2,3,4]

        self.convs = nn.ModuleList()
        for ln in range(self.layer_num):
            if ln == 0:
                self.convs.append(MixHopConv(in_dim_node, hidden_dim, powers=hops))
                if self.use_bn: self.norms.append(nn.BatchNorm1d(hidden_dim*len(hops)))
            else:
                self.convs.append(MixHopConv(hidden_dim*len(hops), hidden_dim, powers=hops))
                if self.use_bn: self.norms.append(nn.BatchNorm1d(hidden_dim*len(hops)))
        
        self.final_project = nn.Linear(hidden_dim*len(hops), out_dim)
        
        self.normalize = utils.torch_normalize_adj
        self.device = device
        
        self.scaling_factor = 2
        self.pseudo_mlp = nn.Sequential(
            nn.Linear(embedding_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_dim, out_dim)) / self.scaling_factor) for i in range(self.layer_num)
        ])
    
    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.final_project.reset_parameters()
        self.pseudo_mlp.reset_parameters()
        self.parsing_mats.reset_parameters()

    def forward_retain(self, x, edge_index, val_test, edge_masks):
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_mask=edge_masks[i])
            if self.use_bn: x = self.norms[i](x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.norms[-1](x)
        # x = self.convs[-1](x, edge_index, edge_mask=edge_weight)  
        
        x = self.final_project(x)
        return x
        
    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])

        edge_weight = None
        if not self.baseline:
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
        self.edge_archive = []
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_mask=edge_weight)
            if self.use_bn: x = self.norms[i](x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.norms[-1](x)
        # x = self.convs[-1](x, edge_index, edge_mask=edge_weight)  
        
        x = self.final_project(x)
        return x
        
    
    def learn_soft_edge3(self, x, edge_index, ln=0):
        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index,  fill_value="mean",
        #             num_nodes=self.num_nodes)
        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        # temp softmax
        # deg = torch.zeros(edge_index.max().item() + 1,
        #                   dtype=torch.float, device=self.device)
        # exp_wei = torch.exp(edge_weight / 3) # Citeseer 1.5 Cora 3
        # deg.scatter_add_(0, edge_index[0], exp_wei)
        # edge_weight = exp_wei / (deg[edge_index[0]] )  # 计算每个边的权重
        return edge_weight




class net_point(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, baseline=False, mode="prune"):
        super().__init__()

        self.mode = mode
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.num_nodes = num_nodes
        self.use_bn = use_bn
        self.use_res = use_res
        self.baseline = baseline
        self.relu = nn.ReLU(inplace=True)
        
        if self.use_bn:
            self.norms = nn.ModuleList()
            for i in range(self.layer_num):
                self.norms.append(nn.BatchNorm1d(embedding_dim[i+1]))
            # print(self.norms)

        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        dropout = dropout
        self.graph_norm = False
        self.batch_norm = False
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.in_channels = in_dim_node
        self.hidden_channels = hidden_dim
        self.out_channels = n_classes
        
        self.initial = MLP(channel_list = [self.in_channels, self.hidden_channels, self.hidden_channels], dropout = self.dropout, norm = None)
        self.pos_gen = MLP(channel_list = [self.hidden_channels, self.hidden_channels, 3], dropout = self.dropout, norm = None)

        self.local_nn = MLP(channel_list = [self.hidden_channels + 3, self.hidden_channels], dropout = self.dropout, norm = None)
        self.global_nn = MLP(channel_list = [self.hidden_channels, self.hidden_channels], dropout = self.dropout, norm = None) 

        self.convs =nn.ModuleList()
        for i in range(self.layer_num):
            self.convs.append(PointNetConv(local_nn = self.local_nn, global_nn = self.global_nn))
        self.final = Linear(in_channels = self.hidden_channels, out_channels = self.out_channels, weight_initializer = 'glorot')


        self.normalize = utils.torch_normalize_adj
        self.device = device
        
        self.scaling_factor = 2
        self.pseudo_mlp = nn.Sequential(
            nn.Linear(embedding_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_dim, out_dim)) / self.scaling_factor) for i in range(self.layer_num)
        ])
    

    def forward_retain(self, x, edge_index, val_test, edge_masks):
        
        x = self.initial(x)
        x_0 = x
        pos = self.pos_gen(x_0)
        for i in range(self.layer_num):
            x = self.convs[i].forward(x = x, edge_index = edge_index, pos = pos, edge_mask=edge_masks[i])
            x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
            x = F.relu(x, inplace = True)
        x = self.final(x)
        return x

    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])

        edge_weight = None
        if (not self.baseline) and (not self.mode == "retain"):
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
        self.edge_archive = []
        
        x = self.initial(x)
        x_0 = x
        pos = self.pos_gen(x_0)
        for i in range(self.layer_num):
            x = self.convs[i].forward(x = x, edge_index = edge_index, pos = pos, edge_mask=edge_weight)
            x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
            x = F.relu(x, inplace = True)
        x = self.final(x)
        return x
    
    def learn_soft_edge3(self, x, edge_index, ln=0):
        
        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        return edge_weight





class net_pgnn(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, 
                 baseline=False, mode="prune", mu=0.1,p=2,K=2,cached=True):
        super().__init__()

        self.mode = mode
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.num_nodes = num_nodes
        self.use_bn = use_bn
        self.use_res = use_res
        self.baseline = baseline
        self.relu = nn.ReLU(inplace=True)
        
        if self.use_bn:
            self.norms = nn.ModuleList()
            for i in range(self.layer_num):
                self.norms.append(nn.BatchNorm1d(embedding_dim[i+1]))

        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        num_heads = 1
        dropout = dropout
        self.graph_norm = False
        self.batch_norm = False
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.lin1 = nn.Linear(in_dim_node, hidden_dim)
        
        self.layers = nn.ModuleList()
        for ln in range(self.layer_num):
            if ln == self.layer_num - 1:
                self.layers.append(pGNNConv(hidden_dim, out_dim, mu, p, K, cached=cached))
            else:
                self.layers.append(pGNNConv(hidden_dim, hidden_dim, mu, p, K, cached=cached))
                    
        self.normalize = utils.torch_normalize_adj
        self.device = device
        
        self.scaling_factor = 2
        self.pseudo_mlp = nn.Sequential(
            nn.Linear(embedding_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_dim, out_dim)) / self.scaling_factor) for i in range(self.layer_num)
        ])
    
    def forward_retain(self, x, edge_index, val_test, edge_masks):
        
        x = self.lin1(x)
        for ln in range(self.layer_num):
            x = self.layers[ln](x, edge_index, edge_mask=edge_masks[ln])
            if ln and self.use_bn: x = self.norms[ln](x)
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if ln and self.use_res: x += h
        return x

    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])

        edge_weight = None
        if (not self.baseline) and (not self.mode == "retain"):
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
        self.edge_archive = []
        
        x = self.lin1(x)
        for ln in range(self.layer_num):
            
            x = self.layers[ln](x, edge_index, edge_mask=edge_weight)
            # print(x.shape)
            if ln and self.use_bn: x = self.norms[ln](x)
            if not ln: h = x
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if ln and self.use_res: x += h
        return x

    def learn_soft_edge3(self, x, edge_index, ln=0):
        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index,  fill_value="mean",
        #             num_nodes=self.num_nodes)
        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        # temp softmax
        # deg = torch.zeros(edge_index.max().item() + 1,
        #                   dtype=torch.float, device=self.device)
        # exp_wei = torch.exp(edge_weight / 3) # Citeseer 1.5 Cora 3
        # deg.scatter_add_(0, edge_index[0], exp_wei)
        # edge_weight = exp_wei / (deg[edge_index[0]] )  # 计算每个边的权重
        return edge_weight



'''
https://github.com/GuanyuCui/MGNN/blob/main/src-graphregression/model.py#L162
'''

class net_mgnn(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, baseline=False, mode='prune', alpha=0.5, beta=0.5, theta=0.5, 
                attention_method = 'concat', initial = 'Linear', eps = 1e-5):
        super(net_mgnn, self).__init__()

        self.mode = mode
        # self.adj_binary = to_dense_adj(edge_index)[0].to(device)
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.num_nodes = num_nodes
        self.use_bn = use_bn
        self.use_res = use_res
        self.baseline = baseline
        
        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]

        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.dropout = dropout
        self.attention_method = attention_method
        self.initial_method = initial
        self.eps = eps

        self.attention = MGNNAttention(in_channels = hidden_dim, hidden_channels = hidden_dim, dropout = self.dropout, method = self.attention_method)
        if self.initial_method == 'Linear':
            self.initial = geo_nn.Linear(in_channels = in_dim_node, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot')
        elif self.initial_method == 'MLP':
            self.initial = geo_nn.MLP(channel_list = [in_dim_node, hidden_dim, hidden_dim], dropout = self.dropout, norm = None)
        elif self.initial_method == 'GC':
            self.initial = GCNConv(in_channels = in_dim_node, out_channels = hidden_dim, bias = True, add_self_loops = False, normalize = False)

        self.convs = nn.ModuleList()
        for i in range(self.layer_num):
            self.convs.append(MGNNConv(channels = hidden_dim, alpha = self.alpha, beta = self.beta, theta = self.theta, layer = i + 1))
        self.final = geo_nn.Linear(in_channels = hidden_dim, out_channels = out_dim, bias = True, weight_initializer = 'glorot')

        self.norm_cache = None
        self.deg_inv_sqrt_cache = None

        self.pseudo_mlp = nn.Sequential(
            nn.Linear(in_dim_node, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        self.scaling_factor = 2
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_dim, out_dim)) / self.scaling_factor) for i in range(self.layer_num)
        ])
        
    def forward_retain(self, x, edge_index, val_test, edge_masks, edge_metric = None):

        row, col = edge_index
        deg = degree(col, x.size(0), dtype = x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        self.norm_cache = (deg_inv_sqrt[row] * deg_inv_sqrt[col])
        self.deg_inv_sqrt_cache = deg_inv_sqrt
        
        # Embedding.
        if self.initial_method == 'GC':
            x = self.initial(x, edge_index)
        else:
            x = self.initial(x)
        x_0 = x
        # Learn edge attention from original data.
        edge_attention = self.attention(x = x_0, edge_index = edge_index)
        # Calculate the edge metric using edge attention and z^{(0)} = f(x).
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]

        edge_metric = (1 - edge_attention) / (1 + edge_attention + self.eps) * torch.norm(x_i - x_j, p = 2, dim = 1)
        # Graph Propagation.
        for i in range(self.layer_num):
            x = self.convs[i].forward(x = x, x_0 = x_0, edge_index = edge_index, edge_metric = edge_metric, norm = self.norm_cache, deg_inv_sqrt = self.deg_inv_sqrt_cache, edge_mask = edge_masks[i])
            x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
            x = F.relu(x, inplace = True)

        x = self.final(x).squeeze()
        return x

    
    def forward(self, x, edge_index, edge_metric = None, val_test=False, **kwargs):
        
        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])

        edge_weight = None
        if not self.baseline:
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
        self.edge_archive = []

        row, col = edge_index
        deg = degree(col, x.size(0), dtype = x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        self.norm_cache = (deg_inv_sqrt[row] * deg_inv_sqrt[col])
        self.deg_inv_sqrt_cache = deg_inv_sqrt
        
        # Embedding.
        if self.initial_method == 'GC':
            x = self.initial(x, edge_index)
        else:
            x = self.initial(x)
        x_0 = x
        # Learn edge attention from original data.
        edge_attention = self.attention(x = x_0, edge_index = edge_index)
        # Calculate the edge metric using edge attention and z^{(0)} = f(x).
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]

        edge_metric = (1 - edge_attention) / (1 + edge_attention + self.eps) * torch.norm(x_i - x_j, p = 2, dim = 1)
        # Graph Propagation.
        for i in range(self.layer_num):
            x = self.convs[i].forward(x = x, x_0 = x_0, edge_index = edge_index, edge_metric = edge_metric, norm = self.norm_cache, deg_inv_sqrt = self.deg_inv_sqrt_cache, edge_mask = edge_weight)
            x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
            x = F.relu(x, inplace = True)

        x = self.final(x).squeeze()
        return x

    def learn_soft_edge3(self, x, edge_index, ln=0):
        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index,  fill_value="mean",
        #             num_nodes=self.num_nodes)
        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        # temp softmax
        # deg = torch.zeros(edge_index.max().item() + 1,
        #                   dtype=torch.float, device=self.device)
        # exp_wei = torch.exp(edge_weight / 3) # Citeseer 1.5 Cora 3
        # deg.scatter_add_(0, edge_index[0], exp_wei)
        # edge_weight = exp_wei / (deg[edge_index[0]] )  # 计算每个边的权重
        return edge_weight
    

    
'''
https://github.com/GitEventhandler/H2GCN-PyTorch/tree/master
'''
class net_h2gcn(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, baseline=False, mode='prune', k = 2, use_relu = True):
        super(net_h2gcn, self).__init__()
        
        self.mode = mode
        # self.adj_binary = to_dense_adj(edge_index)[0].to(device)
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.num_nodes = num_nodes
        self.use_bn = use_bn
        self.use_res = use_res
        self.baseline = baseline
        
        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        
        self.dropout = dropout
        self.act = F.relu if use_relu else lambda x: x
        self.use_relu = use_relu
        self.w_embed = nn.Parameter(
            torch.zeros(size=(in_dim_node, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = nn.Parameter(
            torch.zeros(size=((2 ** (self.layer_num + 1) - 1) * hidden_dim, out_dim)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        
        self.pseudo_mlp = nn.Sequential(
            nn.Linear(in_dim_node, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        self.scaling_factor = 2
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_dim, out_dim)) / self.scaling_factor) for i in range(self.layer_num)
        ])    
    
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)
        # self.pseudo_mlp.reset_parameters()
        # self.parsing_mats.reset_parameters()

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def _eidx_to_sp(self, n, edge_index, device=None):
        indices = edge_index
        values = torch.FloatTensor([1.0] * len(edge_index[0])).to(edge_index.device)
        coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])
        if device is None:
            device = edge_index.device
        return coo.to(device)
    
    def _eidx_to_weight_sp(self, n, edge_index, weight, device=None):
        indices = edge_index
        # values = torch.FloatTensor(weight).to(edge_index.device)
        coo = torch.sparse_coo_tensor(indices=indices, values=weight
                                      , size=[n, n], requires_grad=False)
        if device is None:
            device = edge_index.device
        return coo.to(device)


    def forward_retain(self, x, edge_index, val_test, edge_masks):
        adj = self._eidx_to_sp(len(x), edge_index)
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(x, self.w_embed))]
        for i in range(self.layer_num):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        out = torch.softmax(torch.mm(r_final, self.w_classify), dim=1)
        
        return torch.log(F.softmax(out, dim=1))
    
    
    def forward(self, x, edge_index, val_test=False, **kwargs):
        
        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])
        
        edge_weight = None
        if not self.baseline:
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
        self.edge_archive = []
        
        adj = self._eidx_to_sp(len(x), edge_index)
        weight_matrix = self._eidx_to_weight_sp(len(x), edge_index, edge_weight)
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(x, self.w_embed))]
        for i in range(self.layer_num):
            r_last = rs[-1]
            r1 = torch.sparse.mm(torch.mul(self.a1, weight_matrix), r_last)
            r2 = torch.sparse.mm(torch.mul(self.a2, weight_matrix), r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        out = torch.softmax(torch.mm(r_final, self.w_classify), dim=1)
        
        return torch.log(F.softmax(out, dim=1))
    
    def learn_soft_edge3(self, x, edge_index, ln=0):

        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1

        return edge_weight
    


class net_sgc(nn.Module):
    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, baseline=False, mode="prune"):
        super().__init__()

        self.mode = mode
        self.layer_num = len(embedding_dim) - 1
        
        self.edge_mask_archive = []
        self.num_nodes = num_nodes
        self.use_bn = use_bn
        self.use_res = use_res # TODO
        self.baseline = baseline
        self.relu = nn.ReLU(inplace=True)
        
        if self.use_bn:
            self.norms = nn.ModuleList()
            for i in range(self.layer_num):
                self.norms.append(nn.BatchNorm1d(embedding_dim[i+1]))
            # print(self.norms)

        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        dropout = dropout
        self.graph_norm = False
        self.batch_norm = False
        self.residual = use_res
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.conv = SGConv(
            in_channels=in_dim_node,
            out_channels=out_dim,
            K=self.layer_num,
            cached=True,
        )

        self.normalize = utils.torch_normalize_adj
        self.device = device
        
        self.scaling_factor = 2
        self.pseudo_mlp = nn.Sequential(
            nn.Linear(embedding_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        
        self.parsing_mats = nn.ParameterList([
            nn.Parameter(torch.ones((out_dim, out_dim)) / self.scaling_factor) for i in range(self.layer_num)
        ])
    

    def forward_retain(self, x, edge_index, val_test, edge_masks):
        
        x = self.conv(x, edge_index, edge_mask=edge_masks)
        return x

    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])

        edge_weight = None
        if (not self.baseline) and (not self.mode == "retain"):
            self.edge_weight = edge_weight = self.learn_soft_edge3(x, edge_index)
            print(f"mean: {edge_weight.mean().item()} | std: {edge_weight.std().item()}")
            # print(f"NaN ratio: {torch.isnan(edge_weight).to(torch.float).mean()}")
        self.edge_archive = []
        
        x = self.conv(x, edge_index, edge_mask=edge_weight)
        return x
    
    def learn_soft_edge3(self, x, edge_index, ln=0):
        
        if not ln:
            pseudo_logits = self.pseudo_mlp(x)
            start, end = edge_index
            start_logits, end_logits = pseudo_logits[start], pseudo_logits[end]
            self.outer_products = outer_products = torch.einsum("ec,ed->ecd", start_logits, end_logits)
        parsing = F.relu(self.scaling_factor * self.parsing_mats[ln])
        e = self.outer_products.shape[0]
        ret = torch.bmm(self.outer_products, parsing.unsqueeze(0).expand(e,-1,-1))
        # record the diagonal scores
        diag = torch.diagonal(ret,dim1=-1,dim2=-2)
        self.diag = diag.detach().cpu()
        edge_weight = diag.sum(dim=1)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1
        return edge_weight


from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul


class net_appnp(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, embedding_dim, edge_index, device, num_nodes, dropout, use_res, use_bn, baseline=False, mode="prune", cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(net_appnp, self).__init__(**kwargs)

        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.cached = True

        in_dim_node = embedding_dim[0] # node_dim (feat is an integer)
        hidden_dim = embedding_dim[1]
        out_dim = embedding_dim[-1]
        n_classes = embedding_dim[-1]
        self.num_layers = len(embedding_dim) - 1
        
        self.in_dim = in_dim_node
        self.hid_dim = hidden_dim
        self.out_dim = out_dim
        
        self.device = device
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.use_res = use_res
        self.use_bn = use_bn
        
        self.embedding_dropout = 0.2
        
        self.baseline = baseline
        self.mode = mode
        
        self.alpha=0.1

        self.input_trans = torch.nn.Linear(self.in_dim, self.hid_dim)
        self.output_trans = torch.nn.Linear(self.hid_dim, self.out_dim)
        self.type_norm = 'None' #  if self.dataset != 'ogbn-arxiv' else 'batch'
        if self.type_norm == 'batch':
            self.input_bn = torch.nn.BatchNorm1d(self.hid_dim)
            self.layers_bn = torch.nn.ModuleList([])
            for _ in range(self.num_layers):
                self.layers_bn.append(torch.nn.BatchNorm1d(self.out_dim))

        self.reg_params = list(self.input_trans.parameters())
        self.non_reg_params = list(self.output_trans.parameters())
        if self.type_norm == 'batch':
            for bn in self.layers_bn:
                self.reg_params += list(bn.parameters())

        # self.optimizer = torch.optim.Adam([
        #     dict(params=self.reg_params, weight_decay=self.weight_decay1),
        #     dict(params=self.non_reg_params, weight_decay=self.weight_decay2)
        # ], lr=self.lr)

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward_retain(self, x: Tensor, edge_index: Adj, edge_masks, val_test=False, 
                edge_weight: OptTensor = None,) -> Tensor:
        """"""
        # implemented based on: https://github.com/klicperajo/ppnp/blob/master/ppnp/pytorch/ppnp.py

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # input transformation according to the official implementation
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        x = self.input_trans(x)
        if self.type_norm == 'batch':
            x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        h = self.output_trans(x)
        x = h

        for k in range(self.num_layers):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout, training=self.training)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout, training=self.training)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            if self.type_norm == 'batch':
                x = self.layers_bn[k](x)
            x = x * (1 - self.alpha)
            x += self.alpha * h

        return x

    def forward(self, x: Tensor, edge_index: Adj, val_test=False, 
                edge_weight: OptTensor = None, edge_mask=None, **kwargs) -> Tensor:
        """"""
        # implemented based on: https://github.com/klicperajo/ppnp/blob/master/ppnp/pytorch/ppnp.py

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])
        
        if isinstance(edge_mask, Tensor):
            self.edge_mask = edge_mask
        else:
            self.register_parameter("edge_mask", None)

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # input transformation according to the official implementation
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        x = self.input_trans(x)
        if self.type_norm == 'batch':
            x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        h = self.output_trans(x)
        x = h

        for k in range(self.num_layers):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout, training=self.training)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout, training=self.training)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            if self.type_norm == 'batch':
                x = self.layers_bn[k](x)
            x = x * (1 - self.alpha)
            x += self.alpha * h

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)
    
    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None):

        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim)


    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)