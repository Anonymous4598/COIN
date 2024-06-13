import torch
from abc import ABC, abstractmethod
from torch_scatter import scatter
import torch.nn.functional as F

class Sampler(ABC):
    def __init__(self, intraview_negs=False, intraview_neighbor_negs=False):
        self.intraview_negs = intraview_negs
        self.intraview_neighbor_negs = intraview_neighbor_negs

    # callable() method means the class Sampler can be call, without callable(), the class is only a instance
    def __call__(self, anchor, sample, *args, **kwargs):
        ret = self.sample(anchor, sample, *args, **kwargs)
        if self.intraview_negs: # 
            ret = self.add_intraview_negs(*ret)
        if self.intraview_neighbor_negs:
            ret = self.add_intraview_neighbor_negs(*ret)
        return ret

    @abstractmethod
    def sample(self, anchor, sample, *args, **kwargs):  # abstract method, which needs to be implemented by inherited class
        pass

    @staticmethod
    def add_intraview_negs(anchor, sample, pos_mask, neg_mask):
        num_nodes = anchor.size(0)
        device = anchor.device
        intraview_pos_mask = torch.zeros_like(pos_mask, device=device)
        intraview_neg_mask = torch.ones_like(pos_mask, device=device) - torch.eye(num_nodes, device=device)
        new_sample = torch.cat([sample, anchor], dim=0)                     # (M+N) * K
        new_pos_mask = torch.cat([pos_mask, intraview_pos_mask], dim=1)     # M * (M+N)
        new_neg_mask = torch.cat([neg_mask, intraview_neg_mask], dim=1)     # M * (M+N)
        return anchor, new_sample, new_pos_mask, new_neg_mask
    
    @staticmethod
    def add_intraview_neighbor_negs(anchor, sample, pos_mask, neg_mask, adj):
        num_nodes = anchor.size(0)
        device = anchor.device
        intraview_pos_mask = torch.zeros_like(pos_mask, dtype=torch.float32, device=device)
        adj = adj - torch.diag_embed(adj.diag())
        intraview_pos_mask[adj > 0] = 1.0
        intraview_neg_mask = torch.ones((num_nodes, num_nodes), dtype=torch.float32, device=device) - torch.eye(num_nodes, dtype=torch.float32, device=device) - intraview_pos_mask
        new_sample = torch.cat([sample, anchor], dim=0)
        new_pos_mask = torch.cat([pos_mask, intraview_pos_mask], dim=1)
        new_neg_mask = torch.cat([neg_mask, intraview_neg_mask], dim=1)
        return anchor, new_sample, new_pos_mask, new_neg_mask

class SameScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(SameScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, *args, **kwargs):
        assert anchor.size(0) == sample.size(0)     # 
        num_nodes = anchor.size(0)
        device = anchor.device
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask


class SameScaleNeighborSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(SameScaleNeighborSampler, self).__init__(*args, **kwargs) # tau mean

    def sample(self, anchor, sample, adj, *args, **kwargs): # 
        assert anchor.size(0) == sample.size(0)   # 
        num_nodes = anchor.size(0)  # 
        device = anchor.device
        adj = adj - torch.diag_embed(adj.diag())
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        pos_mask[adj > 0] = 1.0
        neg_mask = torch.ones((num_nodes, num_nodes), dtype=torch.float32, device=device) - pos_mask
        return anchor, sample, pos_mask, neg_mask  # 


class CrossScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(CrossScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, batch=None, neg_sample=None, use_gpu=True, *args, **kwargs):
        num_graphs = anchor.shape[0]  # M
        num_nodes = sample.shape[0]   # N
        device = sample.device

        if neg_sample is not None:
            assert num_graphs == 1  # only one graph, explicit negative samples are needed
            assert sample.shape == neg_sample.shape
            pos_mask1 = torch.ones((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask0 = torch.zeros((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask = torch.cat([pos_mask1, pos_mask0], dim=1)     # M * 2N
            sample = torch.cat([sample, neg_sample], dim=0)         # 2N * K
        else:
            assert batch is not None
            if use_gpu:
                ones = torch.eye(num_nodes, dtype=torch.float32, device=device)     # N * N
                pos_mask = scatter(ones, batch, dim=0, reduce='sum')                # M * N
            else:
                pos_mask = torch.zeros((num_graphs, num_nodes), dtype=torch.float32).to(device)
                for node_idx, graph_idx in enumerate(batch):
                    pos_mask[graph_idx][node_idx] = 1.                              # M * N

        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask


def get_sampler(mode: str, intraview_negs: bool=False, intraview_neighbor_negs: bool=False) -> Sampler:
    if intraview_neighbor_negs:
        if mode in {'L2L', 'C2C', 'G2G'}: # intraview_negs 
            return SameScaleNeighborSampler(intraview_neighbor_negs=intraview_neighbor_negs)
        else:
            return RuntimeError(f'unsupported mode: {mode}')
    else:
        if mode in {'L2L', 'C2C', 'G2G'}:
            return SameScaleSampler(intraview_negs=intraview_negs) 
        elif mode in {'L2G', 'G2L', 'L2C', 'C2L', 'G2C', 'C2G'}:
            return CrossScaleSampler(intraview_negs=intraview_negs)
        else:
            raise RuntimeError(f'unsupported mode: {mode}')
