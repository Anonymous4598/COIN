from typing import *
import os
import sys
import torch
import dgl
import random
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import matplotlib.pyplot as plt


def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    sys.stdout.flush()

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))
       
    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

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

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
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

    features = sp.vstack((allx, tx)).tolil() # 
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) # 
    labels = np.vstack((ally, ty)) # 
    labels[test_idx_reorder, :] = labels[test_idx_range, :] 

    idx_test = test_idx_range.tolist() # ndarray to list
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    # sp.coo.coo_matrix, lil_matrix, np.ndarray, list, list, list
    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
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

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    print("1", type(adj))
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) #np.matrix
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # 
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # 
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # 
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

# coo_matrix 2 sparse.tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
        by using astype(np.float32), the matrix is change into a numpy
        then the sparse_mx.row, sparse_mx.col, sparse_mx.data all ndarray
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32) # only change data from int to float 
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    print("6",shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def seeds_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def draw_energy(V1, V2, V, name, aug_list1, aug_list2, acc1, acc2, acc3, device):
    
    
    plt.figure(figsize=(15,15))
    for ipx in [0,1,2]:

        if ipx == 0:
            adj = V[0]
            X = V[1]
        if ipx == 1:
            adj = V1[0]
            X = V1[1]
        elif ipx==2:
            adj = V2[0]
            X = V2[1]
        num_nodes = X.shape[0]
        indices = torch.LongTensor([adj[0].cpu().numpy(), adj[1].cpu().numpy()]).to(device)
        num_edges = len(adj[0])
        values = torch.FloatTensor(torch.ones(num_edges)).to(device)
        adj_sparse = torch.sparse.FloatTensor(indices, values, torch.Size([num_nodes, num_nodes])).to(device)
        adj = adj_sparse.to_dense()
        X = X.cpu().numpy()
        adj_norm = normalize_adj(adj.cpu())
        adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        A = adj_norm.to_dense().cpu().numpy()

        L_norm = torch.eye(num_nodes) - A
        L = L_norm.numpy()

        import numpy as np
        eigvals, eigvecs = np.linalg.eigh(L)
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)
        # print(np.dot(eigvecs, eigvecs.T))

        idx = np.argsort(eigvals)
        sorted_eigvals = eigvals[idx]
        # assert(sorted_eigvals[0]>=0)
        sorted_eigvecs = eigvecs[:, idx]
        X_hat = np.dot(sorted_eigvecs.T, X)
        UTX = np.dot(sorted_eigvecs.T, X)**2

        # use element-wise log on UTX
        # if args.use_log:
        #     UTX = np.log(UTX + 1e-10)
        # if args.use_mean:
        #     UTX = np.mean(UTX, axis=1, keepdims=True)
        # norms = np.linalg.norm(UTX, ord=1, axis=0, keepdims=True)
        # UTX_normalized = UTX / norms
        # total_sum = np.sum(UTX[:,:]**2)
        fp = []
        plt_pts = []
        lbx = []
        i = 0
        k=0
        while i < len(sorted_eigvals):
            k=i
            while k+1 < num_nodes and sorted_eigvals[k] == sorted_eigvals[k+1]:
                k = k + 1
            lbx.append(sorted_eigvals[k])
            fp.append(np.sum((UTX[:k+1,:])**2))
            i=k+1
        
        # energy distribution function
        total_energy = np.sum((UTX[:,:])**2)
        fp = fp/total_energy
        if ipx == 0:
            plt.plot(lbx, fp, label=f'no-aug')
        elif ipx == 1:
            plt.plot(lbx, fp, label=f'aug1={aug_list1}')
        elif ipx == 2:
            plt.plot(lbx, fp, label=f'aug2={aug_list2}')
    plt.title(f'{name} graph energy distribution V1-V2:{round(acc1, 4)}, V1-V:{round(acc2,4)}, V-V2:{round(acc3,4)}',fontsize=20,fontweight='bold' )
    plt.xlabel('frequency', fontsize=15,fontweight='bold')
    plt.ylabel('graph energy distribution', fontsize=15,fontweight='bold')

    plt.legend(fontsize=18, loc='center',ncol=2, bbox_to_anchor=(0,1,1,0.20))
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.savefig(f'./figs/{name}_{aug_list1}_{aug_list2}.svg',format='svg',dpi=1000)
    plt.clf()
    

def split_dataset(dataset, split_mode, *args, **kwargs):
    assert split_mode in ['rand', 'ogb', 'wikics', 'preload']
    kwargs = kwargs['kwargs']
    if split_mode == 'rand':
        assert 'train_ratio' in kwargs.keys() and 'test_ratio' in kwargs.keys()
        train_ratio = kwargs['train_ratio']
        test_ratio = kwargs['test_ratio']
        num_samples = dataset.x.size(0)
        train_size = int(num_samples * train_ratio)
        test_size = int(num_samples * test_ratio)
        indices = torch.randperm(num_samples)
        
        train_mask = torch.zeros((num_samples,)).to(torch.bool)
        test_mask = torch.zeros((num_samples,)).to(torch.bool)
        val_mask = torch.zeros((num_samples,)).to(torch.bool)

        train_mask[indices[:train_size]] = True
        test_mask[indices[test_size + train_size:]] = True
        val_mask[indices[train_size: test_size + train_size]] = True

        return {
            'train': train_mask,
            'val': val_mask,
            'test': test_mask
        }
    elif split_mode == 'ogb':
        split_res = dataset.get_idx_split()
    elif split_mode == 'wikics':
        assert 'split_idx' in kwargs
        split_idx = kwargs['split_idx']
        # different size with rand split?

        return {
            'train': dataset.train_mask[:, split_idx],
            'test': dataset.test_mask,
            'val': dataset.val_mask[:, split_idx]
        }
        # assert(isinstance(dataset.train_mask, torch.Tensor)==True)
        # return dataset.train_mask, dataset.test_mask, dataset.val_mask
    elif split_mode == 'preload':
        assert 'preload_split' in kwargs
        assert kwargs['preload_split'] is not None
        train_mask, test_mask, val_mask = kwargs['preload_split']

        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }

def get_extra_pos_mask(data):
    # compute extra pos and neg masks for semi-supervised learning
    extra_pos_mask = torch.eq(data.y, data.y.unsqueeze(dim=1)).to('cuda')
    # construct extra supervision signals for only training samples
    extra_pos_mask[~data.train_mask][:, ~data.train_mask] = False
    extra_pos_mask.fill_diagonal_(False)
    # pos_mask: [N, 2N] for both inter-view and intra-view samples
    extra_pos_mask = torch.cat([extra_pos_mask, extra_pos_mask], dim=1).to('cuda')
    # fill interview positives only; pos_mask for intraview samples should have zeros in diagonal
    extra_pos_mask.fill_diagonal_(True)
    return extra_pos_mask

def get_extra_neg_mask(data):
    extra_neg_mask = torch.ne(data.y, data.y.unsqueeze(dim=1)).to('cuda')
    extra_neg_mask[~data.train_mask][:, ~data.train_mask] = True
    extra_neg_mask.fill_diagonal_(False)
    extra_neg_mask = torch.cat([extra_neg_mask, extra_neg_mask], dim=1).to('cuda')
    return extra_neg_mask


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(s):
    return (s.max() - s) / (s.max() - s.mean())


def build_dgl_graph(edge_index: torch.Tensor) -> dgl.DGLGraph:
    row, col = edge_index
    return dgl.graph((row, col))


def batchify_dict(dicts: List[dict], aggr_func=lambda x: x):
    res = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in res:
                res[k] = [v]
            else:
                res[k].append(v)
    res = {k: aggr_func(v) for k, v in res.items()}
    return res

