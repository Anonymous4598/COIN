import argparse
import numpy as np
import scipy.sparse as sp
import torch
import os.path as osp
from datasets import get_dataset
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected
from eval import get_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora')
    args = parser.parse_args()

    torch.cuda.set_device(int(args.device[-1]))
    device = torch.device(args.device)

    path = './data/'
    path = osp.join(path, args.dataset)
    dataset = get_dataset(args.dataset, device=device)
    data = dataset[0]
    data = data.to(device)
    # if args.dataset == 'WiKi-CS':
    #     std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    #     data.x = (data.x - mean) / std
    #     data.edge_index = to_undirected(data.edge_index)

    # else:
    #     split = get_split(num_samples=data.x.size()[0], train_ratio=0.1, test_ratio=0.8)
    #     print(split)

    def coo_adj(ei, ew):
        return to_scipy_sparse_matrix(ei, ew)
    
    def norm_adj(adj):

        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten() 
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. 
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt) 
        norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return norm_adj.A

    def decompose_lap(L):
        eigvals, eigvecs = np.linalg.eigh(L)

        idx = np.argsort(eigvals)
        sorted_eigvals = eigvals[idx]
        sorted_eigvecs = eigvecs[:, idx]
        return sorted_eigvals, sorted_eigvecs

    x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr
    adj = coo_adj(edge_index, edge_weights)
    real_degs = np.sum(adj.A, axis=1, keepdims=True)
    norm_adj = norm_adj(adj)
    L = np.eye(x.size(0)) - norm_adj
    sorted_eigvals ,sorted_eigvecs = decompose_lap(L)
    C = np.transpose(sorted_eigvecs)@x.cpu().numpy()

    norm_A = np.eye(x.size(0)) - sorted_eigvecs@np.diag(sorted_eigvals)@sorted_eigvecs.T
    ones = np.ones((x.size(0), x.size(0)))
    zeros = np.zeros((x.size(0), x.size(0)))
    degs = np.sum(np.where(norm_A>0.01, ones, zeros), axis=1, keepdims=True)

    np.save(f"./data/npy/{args.dataset}_sorted_eigvecs.npy", sorted_eigvecs)
    np.save(f"./data/npy/{args.dataset}_sorted_eigvals.npy", sorted_eigvals)
    np.save(f"./data/npy/{args.dataset}_singal_amps", C)
