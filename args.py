import argparse
import utils
import os, sys
import logging
import glob

def parser_loader():
    parser = argparse.ArgumentParser(description='AdaGLT')
    parser.add_argument('--total_epoch', type=int, default=300)
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument("--retain_epoch", type=int, default=300)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:7")

    parser.add_argument('--model_save_path', type=str, default='model_ckpt',)
    parser.add_argument('--save', type=str, default='CKPTs',
                        help='experiment name')
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--use_bn", action="store_true", default=False)
    parser.add_argument("--use_res", action="store_true", default=False)
    parser.add_argument("--e1", type=float, default=5e-5)
    parser.add_argument("--e2", type=float, default=1e-3)
    parser.add_argument("--coef", type=float, default=0.1)
    parser.add_argument("--task_type", type=str, default="full")
    parser.add_argument("--baseline",action="store_true", default=False)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1e-6)
    parser.add_argument("--model", type=str, default="resgcn")
    parser.add_argument("--pre_mask", type=str)

    args = vars(parser.parse_args())
    seed_dict = {'cora': 1899, 'citeseer': 37899, 'pubmed': 3333}
    if not args['dataset'] in seed_dict.keys():
        seed_dict[args['dataset']] = 2424
        
    # seed_dict = {'cora': 23977/23388, 'citeseer': 27943/27883, 'pubmed': 3333}
    if not args.get("seed"):
        args['seed'] = seed_dict[args['dataset']]

    if args['dataset'] == "cora":
        args['embedding_dim'] = [1433,] +  [64,] * (args['num_layers'] - 1) + [7]
    elif args['dataset'] == "citeseer":
        args['embedding_dim'] = [3703,] +  [64,] * (args['num_layers'] - 1) + [6]
    elif args['dataset'] == "pubmed":
        args['embedding_dim'] = [500,] +  [64,] * (args['num_layers'] - 1) + [3]
    elif args['dataset'] == "Computers":
        args['embedding_dim'] = [767,] +  [64,] * (args['num_layers'] - 1) + [10]
    elif args['dataset'] == "Photo":
        args['embedding_dim'] = [745,] +  [64,] * (args['num_layers'] - 1) + [8]
    elif args['dataset'] == "CS":
        args['embedding_dim'] = [6805,] +  [64,] * (args['num_layers'] - 1) + [15]
    elif args['dataset'] == "Physics":
        args['embedding_dim'] = [8415,] +  [64,] * (args['num_layers'] - 1) + [5]
    elif args['dataset'] == "DBLP":
        args['embedding_dim'] = [1639,] +  [64,] * (args['num_layers'] - 1) + [4]
    elif args['dataset'] == "Chameleon":
        args['embedding_dim'] = [2325,] +  [64,] * (args['num_layers'] - 1) + [5]
    elif args['dataset'] == "Squirrel":
        args['embedding_dim'] = [2089,] +  [64,] * (args['num_layers'] - 1) + [5]
    elif args['dataset'] == "Film":
        args['embedding_dim'] = [932,] +  [64,] * (args['num_layers'] - 1) + [5]
    elif args['dataset'] in ['Texas', 'Cornell', 'Wisconsin']:
        args['embedding_dim'] = [1703,] +  [64,] * (args['num_layers'] - 1) + [5]
    else:
        raise NotImplementedError("dataset not supported.")

    args["model_save_path"] = os.path.join(
        args["save"], args["model_save_path"])
    utils.create_exp_dir(args["save"], scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%m/%d %I:%M:%S %p')
    return args
