from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List
import numpy as np

from functional import dropout_adj, drop_edge_by_weight, get_pagerank_weights, get_degree_weights, get_eigenvector_weights, \
drop_feature, COSTA_feature, add_edge, drop_feature_whole_channel_by_weight, get_pagerank_weights_for_feat, get_degree_weights_for_feat, \
get_eigenvector_weights_for_feat, perturb_edge, dropout_feature, compute_heat, drop_node, compute_markov_diffusion, permute, compute_ppr, random_walk_subgraph 


from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import torch
import scipy.sparse as sp
import time
import os.path as osp


class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, 
            x: torch.FloatTensor,
            edge_index: torch.LongTensor, 
            edge_weight: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(g=Graph(x, edge_index, edge_weight)).unfold()


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


class WeightedChoice(Augmentor):
    def __init__(self, augmentors: List[Augmentor], num_choices: int, aug_p : List):
        super(WeightedChoice, self).__init__()
        self.augmentors = augmentors
        self.num_choices = num_choices
        self.aug_p = aug_p
    
    def augment(self, g:Graph) -> Graph:
        num_augmentors = len(self.augmentors)
        choices = np.random.choice(np.arange(len(self.aug_p)),
                        size=self.num_choices,
                        replace=False,
                        p=self.aug_p)
        for i in choices:
            aug = self.augmentors[i]
            g = aug.augment(g)
        return g

class RandomChoice(Augmentor):
    def __init__(self, augmentors: List[Augmentor], num_choices: int):
        super(RandomChoice, self).__init__()
        assert num_choices <= len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, g: Graph) -> Graph:
        num_augmentors = len(self.augmentors)
        perm = torch.randperm(num_augmentors)
        idx = perm[:self.num_choices]
        for i in idx:
            aug = self.augmentors[i]
            g = aug.augment(g)
        return g



class EdgeRemoving(Augmentor):
    def __init__(self, pe: float, strategy: str = 'uniform'):
        super(EdgeRemoving, self).__init__()
        self.strategy = strategy
        self.pe = pe
    
    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        if self.strategy == 'uniform':
            edge_index = dropout_adj(edge_index, p=self.pe)[0]
        elif self.strategy == 'pr':
            edge_index = drop_edge_by_weight(edge_index, get_pagerank_weights(g)[0], self.pe)
        elif self.strategy == 'evc':
            edge_index = drop_edge_by_weight(edge_index, get_eigenvector_weights(g)[0], self.pe)
        elif self.strategy == 'degree':
            edge_index = drop_edge_by_weight(edge_index, get_degree_weights(g)[0], self.pe) 
        
        return Graph(x=x, edge_index=edge_index, edge_weights=None)

class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=None)

class FeatureCOSTA(Augmentor):
    def __init__(self, pf:float):
        super(FeatureCOSTA, self).__init__()
        self.pf = pf
    
    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = COSTA_feature(x, self.pf) # 
        return Graph(x=x, edge_index=edge_index, edge_weights=None)

class EdgeAdding(Augmentor):
    def __init__(self, pe: float):
        super(EdgeAdding, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index = add_edge(edge_index, ratio=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class EdgeAttrMasking(Augmentor):
    def __init__(self, pf: float, strategy: str = 'uniform'):
        super(EdgeAttrMasking, self).__init__()
        self.pf = pf
        self.strategy = strategy

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        if self.strategy == 'uniform':
            x = drop_feature(x, self.pf)
        elif self.strategy == 'pr':
            x = drop_feature_whole_channel_by_weight(x, get_pagerank_weights_for_feat(g), self.pf) 
        elif self.strategy == 'evc':
            x = drop_feature_whole_channel_by_weight(x, get_eigenvector_weights_for_feat(g), self.pf)
        elif self.strategy == 'degree':
            x = drop_feature_whole_channel_by_weight(x, get_degree_weights_for_feat(g), self.pf)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

class EdgePerturb(Augmentor):
    def __init__(self, pt: float):
        super(EdgePerturb, self).__init__()
        self.pt = pt

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index = perturb_edge(edge_index, ratio=self.pt)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

class FeatureDropout(Augmentor):
    def __init__(self, pf: float):
        super(FeatureDropout, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = dropout_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class HeatDiffusion(Augmentor):
    def __init__(self, temperature: float = 5.0, eps: float = 1e-4, use_cache: bool = True, add_self_loop: bool = True):
        super(HeatDiffusion, self).__init__()
        self.temperature = temperature
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def augment(self, g: Graph) -> Graph:
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = compute_heat(
            edge_index, edge_weights,
            temperature=self.temperature, eps=self.eps, ignore_edge_attr=False, add_self_loop=self.add_self_loop
        )
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res

class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, g: Graph) -> Graph:
        return g

class MarkovDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.05, order: int = 16, sp_eps: float = 1e-4, use_cache: bool = True,
                 add_self_loop: bool = True):
        super(MarkovDiffusion, self).__init__()
        self.alpha = alpha
        self.order = order
        self.sp_eps = sp_eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def augment(self, g: Graph) -> Graph:
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = compute_markov_diffusion(
            edge_index, edge_weights,
            alpha=self.alpha, degree=self.order,
            sp_eps=self.sp_eps, add_self_loop=self.add_self_loop
        )
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res


class NodeDropping(Augmentor):
    def __init__(self, pn: float):
        super(NodeDropping, self).__init__()
        self.pn = pn

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()

        edge_index, edge_weights = drop_node(edge_index, edge_weights, keep_prob=1. - self.pn)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class NodeShuffling(Augmentor):
    def __init__(self):
        super(NodeShuffling, self).__init__()

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = permute(x)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class PPRDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.2, eps: float = 1e-4, use_cache: bool = True, add_self_loop: bool = True):
        super(PPRDiffusion, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def augment(self, g: Graph) -> Graph:
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = compute_ppr(
            edge_index, edge_weights,
            alpha=self.alpha, eps=self.eps, ignore_edge_attr=False, add_self_loop=self.add_self_loop
        )
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res


class RWSampling(Augmentor):
    def __init__(self, num_seeds: int, walk_length: int):
        super(RWSampling, self).__init__()
        self.num_seeds = num_seeds
        self.walk_length = walk_length

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()

        edge_index, edge_weights = random_walk_subgraph(edge_index, edge_weights, batch_size=self.num_seeds, length=self.walk_length)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)



class SubGraph(Augmentor):

    def __init__(self, sub_ratio):
        super(SubGraph, self).__init__()
        self.sub_ratio = sub_ratio

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        num_node = x.size(0)
        sub_num_node = int(num_node * 0.2)
        edge_index_np = edge_index.detach().cpu().numpy()

        idx_sub = [np.random.randint(num_node, size=1)[0]]
        idx_neigh = set([n for n in edge_index_np[1][edge_index_np[0]==idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num_node:
            count = count + 1
            if count > num_node:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh.union(set([n for n in edge_index_np[1][edge_index_np[0]==idx_sub[-1]]]))
        
        idx_drop = [n for n in range(num_node) if not n in idx_sub]
        idx_nondrop = idx_sub
        idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

        edge_index_np = edge_index.detach().cpu().numpy() 

        adj = torch.zeros((num_node, num_node))
        adj[edge_index_np[0], edge_index_np[1]] = 1 
        adj[idx_drop, :] = 0
        adj[:, idx_drop] = 0
        edge_index = torch.nonzero(adj, as_tuple=False).t()

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

class SpecPerb(Augmentor):
    def __init__(self, ds, amplitude, tend, method, mini_batch=False):
        super(SpecPerb, self).__init__()
        self.mini_batch = mini_batch
        self.amplitude = amplitude
        self.tend = tend
        self.ds = ds
        self.method = method 
        
    def coo_adj(self,edge_index):
        return to_scipy_sparse_matrix(edge_index).toarray()
    
    def norm_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)) 
        d_inv_sqrt = np.power(rowsum, -0.5).flatten() 
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. 
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt) 
        norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return norm_adj.A

    def decompose_lap(self, L):
        eigvals, eigvecs = np.linalg.eig(L)
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

        idx = np.argsort(eigvals)
        sorted_eigvals = eigvals[idx]
        sorted_eigvecs = eigvecs[:, idx]
        return sorted_eigvals, sorted_eigvecs

    def augment(self, g: Graph, M=None, Y=None): 
        x, edge_index, edge_weights = g.unfold()
        device = x.device
        adj = to_scipy_sparse_matrix(edge_index)

        rowsum = np.array(adj.sum(1)) 
        d_inv_sqrt = np.power(rowsum, -0.5).flatten() 
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. 
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt) 
        norm_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        sp_L = sp.eye(x.size(0)) - norm_adj
        eigvecs, eigvals, C = None, None, None
        if self.mini_batch == False:
            if osp.exists(f'/home/linzy/code/GSSL/MyGCL/npy/{self.ds}_sorted_eigvecs.npy'):
                eigvecs = np.load(f"/home/linzy/code/GSSL/MyGCL/npy/{self.ds}_sorted_eigvecs.npy")
            if osp.exists(f'/home/linzy/code/GSSL/MyGCL/npy/{self.ds}_sorted_eigvals.npy'):
                eigvals = np.load(f"/home/linzy/code/GSSL/MyGCL/npy/{self.ds}_sorted_eigvals.npy")
            if osp.exists(f'/home/linzy/code/GSSL/MyGCL/npy/{self.ds}_singal_amps.npy'):
                C = np.load(f"/home/linzy/code/GSSL/MyGCL/npy/{self.ds}_singal_amps.npy")
        if eigvecs is None or eigvals is None:
            eigvals, eigvecs = np.linalg.eigh(sp_L.A)
        idx = np.argsort(eigvals)
        sorted_eigvals = eigvals[idx]
        sorted_eigvecs = eigvecs[:, idx]
        
        if C is None:
            C = sorted_eigvecs.T@x.cpu().numpy()  

        method1, method2 = self.method.split(",")
        st = time.time()

        if Y is not None: 
            Y = np.tile(np.subtract(1,Y).reshape(-1,1), (1, x.size(1)))
        else:
            if method2 == "random": 
                Y = np.tile(np.random.normal(loc=0, scale=1, size=x.size(0)).reshape(-1,1), (1, x.size(1)))
            elif method2 == "rebalance": 
                sampled_y = np.random.normal(loc=0, scale=1, size=x.size(0)).reshape(-1,1)
                B = np.sum(C**2, axis=1, keepdims=True)**(1/2)
                D = np.array(sampled_y) * np.array(B)
                D = D**2
                D = D/np.sum(D)
                Y = np.tile(np.subtract(1, D), (1, x.size(1))) 
            elif method2 == "rebalance2": 
                sampled_y = np.random.normal(loc=0, scale=1, size=x.size(0)*x.size(1)).reshape(x.size(0), -1)
                print("sampled_y\n", sampled_y)
                print("original C\n", C)
                D = np.abs((np.array(sampled_y) * np.array(C)))
                print("D\n", D)
                E = np.sum(D, axis=0, keepdims=True)
                print("E\n", E)
                D = D/E
                Y = np.subtract(1,D)
                print("Y\n", Y)
            elif method2 == "none":
                Y = np.ones((C.shape[0], C.shape[1]))
        st =time.time()
        C_perb = C*Y 
        print("after perb, the C is\n", C_perb)
        x = torch.from_numpy(sorted_eigvecs@C_perb)
        print("after perb, the x is\n", x)
        x = torch.tensor(x, dtype=torch.float32, requires_grad=False).to(device)

        st =time.time()

        if M is not None:
            specperb_eigvals = np.array(M) * np.array(sorted_eigvals)
        else:
            if method1 == 'linear':
                perbs = np.linspace(0, self.amplitude, x.size(0))
                if self.tend == 'dec':
                    for i, t in np.ndenumerate(perbs):
                        t = (1-t)
                        perbs[i] = t

                elif self.tend == 'inc':
                    for i, t in np.ndenumerate(perbs):
                        t = (1+t)
                        perbs[i] = t
                specperb_eigvals = np.array(perbs) * np.array(sorted_eigvals)
                for i, t in np.ndenumerate(specperb_eigvals):
                    if t < 0:
                        specperb_eigvals[i] = abs(0.0)
                    if t > 2:
                        specperb_eigvals[i] = 2.0
                specperb_eigvals = np.diag(specperb_eigvals)
            elif method1 == 'uniform':
                perb_matrix = np.diag(np.random.uniform(low=1-self.amplitude, high=1, size=x.size(0)))
                specperb_eigvals = perb_matrix@np.diag(sorted_eigvals)
            elif method1 == 'rebalance':
                perb_matrix = np.diag(np.random.normal(loc=0, scale=1, size=x.size(0)))
                perb_matrix = perb_matrix@sorted_eigvals
                op_str = 'square'
                if op_str == 'square':
                    perb_matrix = perb_matrix ** 2
                elif op_str == 'exp':
                    tau = 0.3
                    f = lambda x : torch.exp(x / tau)
                    perb_matrix = f(perb_matrix)
                perb_matrix = perb_matrix / np.trace(perb_matrix)
                perb_matrix = np.eye(x.size(0)) - perb_matrix
                specperb_eigvals = perb_matrix@np.diag(sorted_eigvals)
            elif method1 == 'corruption':
                indices = torch.randperm(x.size(0))
                specperb_eigvals = np.diag(sorted_eigvals[indices])
            elif method1 == 'none':
                specperb_eigvals = np.diag(sorted_eigvals)
        st=time.time()
        L_perb = sorted_eigvecs@specperb_eigvals@sorted_eigvecs.T
        A_perb_norm = np.eye(x.size(0)) - L_perb
        A_sp_perb = sp.coo_matrix(A_perb_norm)

        edge_index, edge_weights = from_scipy_sparse_matrix(A_sp_perb)
        edge_index = torch.tensor(edge_index, dtype=torch.int64, requires_grad=False).to(device)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32, requires_grad=False).to(device)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
