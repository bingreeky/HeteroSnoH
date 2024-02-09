import os
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import is_undirected, to_undirected
import net as net
import layers
from args import parser_loader
import utils
from sklearn.metrics import f1_score
import pdb
import pruning
import copy
from scipy.sparse import coo_matrix
import warnings
warnings.filterwarnings('ignore')



def run_get_mask(args):
    device = args['device']
    
    adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type = \
        utils.load_citation(args['dataset'], task_type=args['task_type'])  # adj: csr_matrix
    if adj.shape[0] != 2:
        adj = adj.to_dense().to(device).to(torch.float32)
        adj = adj.nonzero().t().contiguous()
    else:
        adj = adj.to(device)
    features = features.to(device).to(torch.float32)
    labels = labels.to(device)
    loss_func = nn.CrossEntropyLoss()
    

            
    # if args['model'] in ['gat','gcn2','mixhop']:
    adj, _ = remove_self_loops(adj)
    adj, _ = add_self_loops(adj,  fill_value="mean",
                num_nodes=features.shape[0])

    if not is_undirected(adj):
        adj = to_undirected(adj)
    
    net_gcn = net.net_selector(args['model'])(embedding_dim=args['embedding_dim'], edge_index=adj, device=device, 
                                num_nodes=features.shape[0], dropout=args['dropout'], use_bn=args['use_bn'], use_res=args['use_res'],baseline=args['baseline'])
    net_gcn = net_gcn.to(device)

    optimizer = torch.optim.Adam(net_gcn.parameters(
    ), lr=args['lr'], weight_decay=args['weight_decay'])

    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "adj_spar": 0, "wei_spar": 0}
    best_target = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "mask": None}
    
    performance_ls = {"test":[],"val":[],"train":[]}
    
    # rewind_weight = copy.deepcopy(net_gcn.state_dict())
    for epoch in range(args['total_epoch']): 
        net_gcn.train()

        optimizer.zero_grad()
        output = net_gcn(features, adj)

        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        
        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            output = net_gcn(features, adj, val_test=True)
            acc_val = f1_score(labels[idx_val].cpu().numpy(
            ), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(
            ), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            acc_train = f1_score(labels[idx_train].cpu().numpy(
            ), output[idx_train].cpu().numpy().argmax(axis=1), average='micro')

            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                # best_weight = copy.deepcopy(torch.concat(net_gcn.edge_archive))
                # best_weight = copy.deepcopy(net_gcn.edge_weight.detach().cpu()) # (E,)
                if not args['baseline']: best_weight = net_gcn.edge_weight
                # best_mask = [copy.deepcopy(net_gcn.edge_mask_archive), copy.deepcopy(net_gcn.generate_wei_mask())]

            print("Epoch:[{}] L:[{:.3f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] |"
                  .format(epoch, loss.item(), acc_train * 100, acc_val * 100, acc_test * 100, ), end=" ")
            print("Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                    .format(
                        best_val_acc['val_acc'] * 100,
                        best_val_acc['test_acc'] * 100,
                        best_val_acc['epoch']))
            performance_ls['val'].append(acc_val)
            performance_ls['test'].append(acc_test)
            performance_ls['train'].append(acc_train)
    # for key in ['val','test','train']:
    scp_arr = np.array(performance_ls['test'])
    np.save(f"./train_vis2/{args['dataset']}_{args['model']}_{args['num_layers']}_{args['beta']}_baseline.npy",scp_arr)
    

    return adj, best_weight, (adj, features, labels, idx_train, idx_val, idx_test)

def run_fix_mask(args, edge_masks, hoho=None):# TODO change hoho default
    device = args['device']

    edge_masks = [mask.to(device) for mask in edge_masks]
    print(edge_masks[0].shape)
    print(edge_masks[0].requires_grad)
    
    # print(edge_masks)
    # adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type = \
    #     utils.load_citation(args['dataset'], task_type=args['task_type'])  # adj: csr_matrix
    if hoho:
        adj, features, labels, idx_train, idx_val, idx_test = hoho
    else:
        adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type = \
            utils.load_citation(args['dataset'], task_type=args['task_type'])  # adj: csr_matrix
        # print("adfer loading...")
        # print(adj.shape)
    if adj.shape[0] != 2:
        adj = adj.to_dense().to(device).to(torch.float32)
        adj = adj.nonzero().t().contiguous()
    else:
        adj = adj.to(device)
    features = features.to(device).to(torch.float32)
    labels = labels.to(device)
    loss_func = nn.CrossEntropyLoss()
    
    # if not args['model'] in ['gcn','resgcn']:
    adj, _ = remove_self_loops(adj)
    adj, _ = add_self_loops(adj,  fill_value="mean",
                num_nodes=features.shape[0])

    if not is_undirected(adj):
        adj = to_undirected(adj)
    
    net_gcn = net.net_selector(args['model'])(embedding_dim=args['embedding_dim'], edge_index=adj, device=device,
                                num_nodes=features.shape[0], dropout=args['dropout'],
                                use_bn=args['use_bn'], use_res=args['use_res'], mode="retain")
    net_gcn = net_gcn.to(device)

    optimizer = torch.optim.Adam(net_gcn.parameters(
    ), lr=args['lr'], weight_decay=args['weight_decay'])

    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}
    
    performance_ls = {"test":[],"val":[],"train":[]}
    
    for epoch in range(args['retain_epoch']):
        net_gcn.train()
        optimizer.zero_grad()
        output = net_gcn(features, adj, edge_masks=edge_masks)

        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        
        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            # print(edge_masks[-1].sum()/edge_masks[-1].shape[0])
            output = net_gcn(features, adj, val_test=True,
                            edge_masks=edge_masks)
            acc_val = f1_score(labels[idx_val].cpu().numpy(
            ), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(
            ), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            acc_train = f1_score(labels[idx_train].cpu().numpy(
            ), output[idx_train].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch

            print("Epoch:[{}] L:[{:.3f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                  .format(epoch, loss.item() ,acc_train * 100 ,acc_val * 100, acc_test * 100,
                          best_val_acc['val_acc'] * 100,
                          best_val_acc['test_acc'] * 100,
                          best_val_acc['epoch']))
            performance_ls['val'].append(acc_val)
            performance_ls['test'].append(acc_test)
            performance_ls['train'].append(acc_train)
    for key in ['val','test','train']:
        scp_arr = np.array(performance_ls[key])
        np.save(f"./train_vis2/{args['dataset']}_{args['model']}_{args['num_layers']}_{args['beta']}_{key}.npy",scp_arr)
    # return best_val_acc['test_acc']

if __name__ == "__main__":

    args = parser_loader()
    print(args)
    # torch.autograd.set_detect_anomaly(True)
    utils.fix_seed(args['seed'])
    
    
    if args['pre_mask']:
        edge_masks = np.load(args['pre_mask'])
        edge_masks = [mask for mask in  torch.from_numpy(edge_masks)]
        run_fix_mask(args, edge_masks)
    
    edge_index, edge_score, hoho = run_get_mask(args)
    
    edge_masks = utils.construct_sparse_graph_torch3(
        edge_index, edge_score, args, ratio=args['beta'], \
        manual_remain_loop=True ,dense=False)
    run_fix_mask(args, edge_masks, hoho)