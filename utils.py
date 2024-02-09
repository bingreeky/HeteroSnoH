import math
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh
from typing import Optional
import random
import sys
import os
import pdb
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import subgraph, k_hop_subgraph, to_dense_adj
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected
from torch_geometric.datasets import Planetoid, CoraFull, CitationFull, Amazon, Coauthor, WebKB, WikipediaNetwork, Actor, DeezerEurope, WikiCS, LINKXDataset
from torch_geometric.utils import to_torch_coo_tensor, dense_to_sparse
from torch_geometric.data import Data
import shutil
import layers
import scipy.optimize as optimize
from tqdm import tqdm

# from dgl.data import CoraGraphDataset, KarateClubDataset, CiteseerGraphDataset, PubmedGraphDataset
# from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split

from normalization import fetch_normalization, row_normalize


datadir = "data"


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # preprocess feature
    features = preprocess_features(features)
    features = torch.tensor(features, dtype=torch.float32)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # preprocess adj
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
    # adj = torch_normalize_adj(adj)
    # adj2 = preprocess_adj(adj)
    # adj2 = sparse_mx_to_torch_sparse_tensor(adj2).to_dense()
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    _, l_num = labels.shape
    labels = torch.tensor((labels * range(l_num)).sum(axis=1), dtype=torch.int64)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y)+500))

    print(f"train: {len(idx_test)} val: {len(idx_val)} test: {len(idx_test)}")

    return adj, features, labels, idx_train, idx_val, idx_test


def load_hetedata(dataset_name, root_dir="./data", split_ratio=[0.6,0.2,0.2], self_loops=False):
    graph_adjacency_list_file_path = os.path.join(root_dir, dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join(root_dir, dataset_name,
                                                                f'out1_node_feature_label.txt')
    graph_node_features_dict = {}
    graph_labels_dict = {}
    edge_index = []
    if dataset_name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])
                    
    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)       
            edge_index.append([int(line[0]), int(line[1])])
    num_node = len(graph_labels_dict)
    data_edge_index = torch.LongTensor(edge_index).T            
    node_label = torch.LongTensor([graph_labels_dict[i] for i in range(num_node)])
    node_feature = np.array([graph_node_features_dict[i] for i in range(num_node)])
    node_feature = torch.Tensor(preprocess_features(node_feature))
    
    graph = Data(x=node_feature, edge_index=data_edge_index, y=node_label)
    
    num_features = graph.num_features
    num_classes = int(node_label.max().item()+1)
    num_nodes = graph.x.size(0)
    
    if self_loops:
        graph.edge_index = add_self_loops(graph.edge_index)[0]
    else:
        graph.edge_index = remove_self_loops(graph.edge_index)[0]

    if split_ratio is not None:
        if np.array(split_ratio).sum() != 1:
            raise Exception("split_ratio must sum to 1")
        else:
            ids = torch.randperm(num_nodes)
            s1 = int(num_nodes * split_ratio[0])
            s2 = int(num_nodes * (split_ratio[0]+split_ratio[1]))
            split_idx = {
                "train": ids[:s1],
                "valid": ids[s1:s2],
                "test": ids[s2:]
            }
    else:
        split_idx = {
            "train": torch.nonzero(graph.train_mask).squeeze(1),
            "valid": torch.nonzero(graph.val_mask).squeeze(1),
            "test": torch.nonzero(graph.test_mask).squeeze(1),
        }

    train_num, val_num, test_num = split_idx["train"].size(0), split_idx["valid"].size(0), split_idx["test"].size(0)
    print(f"train num:{train_num}, valid num:{val_num},test num:{test_num}")

    return graph.edge_index,  node_label, split_idx['train'], split_idx['valid'], split_idx['test'], [], []

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    #return sparse_to_tuple(features)
    if isinstance(features, np.ndarray):
        return features
    return features.todense()


def torch_normalize_adj(adj, device):
    # adj = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to(device)
    return adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    #return sparse_to_tuple(adj_normalized)
    return adj_normalized


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


# def chebyshev_polynomials(adj, k):
#     """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
#     print("Calculating Chebyshev polynomials up to order {}...".format(k))

#     adj_normalized = normalize_adj(adj)
#     laplacian = sp.eye(adj.shape[0]) - adj_normalized
#     largest_eigval, _ = eigsh(laplacian, 1, which='LM')
#     scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

#     t_k = list()
#     t_k.append(sp.eye(adj.shape[0]))
#     t_k.append(scaled_laplacian)

#     def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
#         s_lap = sp.csr_matrix(scaled_lap, copy=True)
#         return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

#     for i in range(2, k+1):
#         t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

#     return sparse_to_tuple(t_k)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_adj_raw(dataset_str):

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    adj_raw = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj_raw

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def get_dataset(name:str, root:str="data/" ):
    # root += name
    if name in ['Computers', 'Photo']:
        dataset = Amazon(root=root, name=name, transform=T.NormalizeFeatures())
        data = dataset[0]

    elif name in ['CS', 'Physics']:
        dataset = Coauthor(root=root, name=name, transform=T.NormalizeFeatures())
        data = dataset[0]

    elif name in ['DBLP']:
        dataset = CitationFull(root=root, name=name, transform=T.NormalizeFeatures())
        data = dataset[0]

    elif name in ['Chameleon', 'Squirrel']:
        preProcDs = WikipediaNetwork(
            root=root, name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=root, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index

    elif name in ['Film']:
        dataset = Actor(root=root, transform=T.NormalizeFeatures())
        data = dataset[0]

    elif name in ['Texas', 'Cornell', 'Wisconsin']:
        dataset = WebKB(root=root, name=name, transform=T.NormalizeFeatures())
        data = dataset[0]

    elif name in ['Penn94', 'Cornell5']:
        dataset = LINKXDataset(root=root, name=name)
        data = dataset[0]
        
    num_nodes = data.y.shape[0]
    all_idx = np.array(range(num_nodes))
    dataset_split = [0.6, 0.2, 0.2] # train/val/test
    np.random.shuffle(all_idx)
    data.train_idx = torch.LongTensor(all_idx[:int(num_nodes*dataset_split[0])])
    data.val_idx = torch.LongTensor(all_idx[int(num_nodes*dataset_split[0]):int(num_nodes*dataset_split[0])+int(num_nodes*dataset_split[1])])
    data.test_idx = torch.LongTensor(all_idx[int(num_nodes*dataset_split[0])+int(num_nodes*dataset_split[1]):])
    print(f"train: {len(data.train_idx)} nodes | val: {len(data.val_idx)} nodes | test: {len(data.test_idx)} nodes")
    return data.edge_index, data.x, data.y, data.train_idx, data.val_idx, data.test_idx, [] ,[]

def load_citation(dataset_str="cora", normalization="AugNormAdj", porting_to_torch=True,data_path=datadir, task_type="semi"):
    """
    Load Citation Networks Datasets.
    """
    if not dataset_str in ['cora','pubmed','citeseer']:
        return get_dataset(dataset_str)
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # degree = np.asarray(G.degree)
    degree = np.sum(adj, axis=1)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    if task_type == "full":
        print("Load full supervised task.")
        #supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally)- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        #semi-supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")

    adj, features = preprocess_citation(adj, features, normalization)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=1)
    # porting to pytorch
    if porting_to_torch:
        features = torch.FloatTensor(features).float()
        labels = torch.LongTensor(labels)
        # labels = torch.max(labels, dim=1)[1]
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        degree = torch.LongTensor(degree)
    learning_type = "transductive"
    
    print(f"train: {len(idx_train)} val: {len(idx_val)} test: {len(idx_test)}")
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type

def sgc_precompute(features, adj, degree):
    #t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = 0 #perf_counter()-t
    return features, precompute_time

def fix_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    
def print_sparsity(edge_masks, edge_num):
    with torch.no_grad():
        spar_ls = []
        for ln in range(len(edge_masks)):
            spar = (1 - (edge_masks[ln].sum().item() / edge_num)) * 100
            spar_ls.append(spar)
            print(f"layer {ln}: [{spar:.4f}%]")    
        print("="*20)
        print(f"avg sparsity: [{np.mean(spar_ls):.4f}%]")
        print("="*20)

def judge_spar(spar, target):
    return spar >= (target - 2) and spar <= (target + 2)


def calcu_sparsity(edge_masks, edge_num):
    if edge_masks is None:
        return 0
    with torch.no_grad():
        spar_ls = []
        for ln in range(len(edge_masks)):
            spar = (1 - (edge_masks[ln].sum().item() / edge_num)) * 100
            spar_ls.append(spar)
        return np.mean(spar_ls)



def denominator(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    # ks_sum refers to O_k in the paper
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones)
    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)


def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / \
        torch.sqrt(torch.sqrt(torch.tensor(
            data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / \
        torch.sqrt(torch.tensor(
            projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash,
                      dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                                                        dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    return data_dash


def degree(index: torch.Tensor, num_nodes: Optional[int] = None,
           dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:

        >>> row = torch.tensor([0, 1, 0, 2, 0])
        >>> degree(row, dtype=torch.long)
        tensor([3, 1, 1])
    """
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(
                    path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)


def row_normalize_adjacency_matrix(adj_matrix):
    device = adj_matrix.device
    # Calculate the degree matrix D by summing along the rows of the adjacency matrix
    degree_matrix = torch.diag(1. / torch.sum(adj_matrix, dim=1))
    
    # Calculate the inverse of the degree matrix
    degree_inv_matrix = degree_matrix.masked_fill_(degree_matrix == float('inf'), 0)
    degree_inv_matrix.masked_fill_(degree_inv_matrix.isnan(), 0)

    # with torch.no_grad():
    #     print(f"[{(degree_inv_matrix.isnan()).sum().item()}]")
    #     print(f"[{(degree_inv_matrix == float('inf')).sum().item()}]")
    
    # Compute the normalized adjacency matrix A_norm = -D^{-1} A
    normalized_adj_matrix = torch.mm(degree_inv_matrix, adj_matrix)
    

    # zero_row_indices = torch.where(normalized_adj_matrix.sum(dim=1) == 0)[0]
    # normalized_adj_matrix[zero_row_indices, zero_row_indices] = 1
    # return torch.eye(adj_matrix.shape[0]).to(device) - normalized_adj_matrix
    return  normalized_adj_matrix


@torch.no_grad()
def net_weight_sparsity(model: nn.Module):
    total, keep = 0., 0.
    for layer in model.modules():
        if isinstance(layer, layers.MaskedLinear):
            abs_weight = torch.abs(layer.weight)
            threshold = layer.threshold.view(abs_weight.shape[0], -1)
            abs_weight = abs_weight-threshold
            mask = layer.step(abs_weight)
            ratio = torch.sum(mask) / mask.numel()
            total += mask.numel()
            keep += torch.sum(mask)
            # logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
            # logger.info("{}, keep ratio {:.4f}".format(layer, ratio))
    if not total:
        return 0
    else:
        return float(1 - keep / total) * 100
    
def initalize_thres(coef):
    def equation(x):
        return x**3 + 20*x + 0.2/coef
    
    result = optimize.root_scalar(equation, bracket=[-10, 10], method='bisect')
    return result.root
    
    
def construct_sparse_graph(edge_index, edge_weight, num_layers):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    if isinstance(edge_weight, torch.Tensor):
        edge_weight = edge_weight.detach().cpu().numpy()
    G = nx.Graph()
    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        start_node = edge_index[0, i]
        end_node = edge_index[1, i]
        weight = edge_weight[i]
        G.add_edge(start_node, end_node, weight=weight)
    
    node_hop_ls = []
    for n in tqdm(G.nodes): 
        score_ls = []
        for l in range(num_layers):
            if not l: score_ls.append(1)
            subgraph = nx.ego_graph(G,n,radius=num_layers)
            neighbors= list(subgraph.nodes())
            avg_score = edge_weight[neighbors].mean()
            score_ls.append(avg_score)
        node_hop_ls.append(score_ls)
    node_hop_ls = np.array(node_hop_ls)  
    np.save("node_hop_ls.npy", node_hop_ls, allow_pickle=True)
    
@torch.no_grad()
def construct_sparse_graph_torch2(edge_index, edge_weight, args, dense=False):
    num_layers, num_classes = args['num_layers'], args['embedding_dim'][-1]
    # edge weight (E,3)
    num_edges = edge_index.shape[1]
    num_nodes = edge_index.max().item() + 1
    edge_index = edge_index.to("cpu")
    edge_weight = edge_weight.to("cpu") # (E,C)
    src_nodes = edge_index[0].numpy()
    # print(edge_weight.mean(), edge_weight.std(), edge_weight.nonzero())
    adj_ls = []
    for cl in range(edge_weight.shape[1]):
        adj_ls.append(to_dense_adj(edge_index, edge_attr=edge_weight[:,cl])[0])
    adj_ls = torch.stack(adj_ls, dim=0)
    np.save("adj_ls.npy", adj_ls.numpy(), allow_pickle=True)
    
    for i in range(len(adj_ls)):
        adj_ls[i] /= adj_ls[i].sum(axis=1,keepdims=True)
    
    adjs = adj_ls
    node_hop_ls = []
    for k in (range(num_layers)):
        adj_k_ls = [] # C, N, N
        score_k = []
        for cl in range(num_classes):
            adj_k_ls.append(adjs[cl]**k)
        for a in adj_k_ls:
            score_k.append(np.ma.masked_equal(a, 0).mean(axis=1).data)
        score_k = np.stack(score_k).mean(axis=0)
        node_hop_ls.append(score_k)
    node_hop_ls = np.stack(node_hop_ls).T
    np.save("node_hop_ls.npy", node_hop_ls)
    if np.isnan(node_hop_ls).any():
        raise scddvgesr
    
    ratio = 1e-5
    result_indices = np.empty(num_nodes, dtype=int)
    for i in range(num_nodes):
        first_value = node_hop_ls[i, 1]
        last_greater_index = -1
        for j in range(2, num_layers):
            if node_hop_ls[i, j] > first_value * ratio:
                last_greater_index = j
        result_indices[i] = last_greater_index
    print(result_indices)
    node_masks = np.ones(shape=(num_nodes, num_layers),dtype=bool)
    for i in range(num_nodes):
        last_greater_index = result_indices[i]
        if last_greater_index >= 0:
            node_masks[i, (last_greater_index+1):] = 0
    np.save("node_masks.npy", node_masks)
    
    for i in range(node_masks.shape[1]):
        print(f"layer {i}: ", node_masks[:,i].sum(), "nodes")
    
    edge_masks = []
    node_ls = np.arange(num_nodes)
    for ln in range(num_layers):
        edge_masks.append(np.isin(src_nodes, node_ls[node_masks[:,ln]]))
    edge_masks = np.stack(edge_masks)
    np.save("edge_masks.npy", edge_masks)
    
    for i in range(edge_masks.shape[0]):
        print(f"layer {i}: ", edge_masks[i].sum(), "edges")
    
    
    # print( to_dense_adj(edge_index,edge_attr=torch.from_numpy(edge_masks)[0]).shape)
    if not dense:
        return [mask for mask in torch.from_numpy(edge_masks)]
    else: 
        edge_masks = [to_dense_adj(edge_index,edge_attr=mask)[0] for mask in torch.from_numpy(edge_masks)]
        # print(edge_masks[0].sum())
        # print(edge_masks[-1].sum())
        return edge_masks

@torch.no_grad()
def construct_sparse_graph_torch3(edge_index, edge_weight, args, ratio, dense=False, manual_remain_loop=False):
    # edge weight: (E,)
    num_layers, num_classes = args['num_layers'], args['embedding_dim'][-1]
    num_nodes = edge_index.max().item() + 1
    edge_index = edge_index.to("cpu")
    edge_weight = edge_weight.to("cpu")
    src_nodes = edge_index[0].numpy()
    print("creating and normalizing adjacency matrix...")
    adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    adj = torch_normalize_adj(adj, adj.device) # .numpy()
    edge_masks, score_k, node_hop_ls = [], [], []
    print("processing adjacency matrix...")
    adj = adj.to(args['device'])
    for ln in tqdm(range(num_layers)):
        adj_k = adj**ln # n layers->n hops
        # TODO try not scale adj
        score_k.append(np.ma.masked_equal(adj_k.cpu().numpy(), 0).mean(axis=1).data)
        # score_k: (N,)
    node_hop_ls = np.stack(score_k).T
    np.save("node_hop_ls.npy", node_hop_ls)
    if np.isnan(node_hop_ls).any():
        raise ValueError
    
    # ratio = 1e-6 # pubmed 5e-3
    result_indices = np.empty(num_nodes, dtype=int)
    print("calculating heterophily threshold for each node...")
    for i in tqdm(range(num_nodes)):
        first_value = node_hop_ls[i, 0] # TODO
        last_greater_index = 1
        for j in range(2, num_layers): # TODO
            if node_hop_ls[i, j] > first_value * ratio:
                last_greater_index = j
        result_indices[i] = last_greater_index
    print(result_indices)
    node_masks = np.ones(shape=(num_nodes, num_layers),dtype=bool)
    for i in range(num_nodes):
        last_greater_index = result_indices[i]
        if last_greater_index >= 0:
            node_masks[i, (last_greater_index+1):] = 0
    np.save("node_masks.npy", node_masks)
    
    for i in range(node_masks.shape[1]):
        print(f"layer {i}: ", node_masks[:,i].sum(), "nodes")
    
    edge_masks = []
    node_ls = np.arange(num_nodes)
    for ln in range(num_layers):
        edge_masks.append(np.isin(src_nodes, node_ls[node_masks[:,ln]]))
    edge_masks = np.stack(edge_masks)
    np.save("edge_masks.npy", edge_masks)
    
    for i in range(edge_masks.shape[0]):
        print(f"layer {i}: ", edge_masks[i].sum(), "edges")
    
    
    # print( to_dense_adj(edge_index,edge_attr=torch.from_numpy(edge_masks)[0]).shape)
    if not dense:
        edge_masks = [mask for mask in  torch.from_numpy(edge_masks)]
        if manual_remain_loop:
            self_loop_indices = (edge_index[0] == edge_index[1])
            for i in range(len(edge_masks)):
                edge_masks[i][self_loop_indices] = True
        # for i in range(len(edge_masks)):
        #     index, value = dense_to_sparse(edge_masks[i])
        #     edge_masks[i] = to_torch_coo_tensor(index, value)
    else: 
        edge_masks = [to_dense_adj(edge_index,edge_attr=mask)[0] for mask in torch.from_numpy(edge_masks)]
        for i in range(len(edge_masks)):
            index, value = dense_to_sparse(edge_masks[i])
            edge_masks[i] = to_torch_coo_tensor(index, value)
        if manual_remain_loop:
            self_loop_indices = (edge_index[0] == edge_index[1])
            for i in range(len(edge_masks)):
                edge_masks[i].values()[self_loop_indices] = True
    return edge_masks

        
    
    
    
    