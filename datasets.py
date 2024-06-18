from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset
import torch
import os
import os.path as osp
import numpy as np

dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M 
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M 
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}

splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N', 
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_', 
}

class BaseGraph:
    '''
        A general format for datasets.
        Args:
            x (Tensor): node feature, of shape (number of node, F).
            edge_index (LongTensor): of shape (2, number of edge)
            edge_weight (Tensor): of shape (number of edge)
            mask: a node mask to show a training/valid/test dataset split, of shape (number of node). mask[i]=0, 1, 2 means the i-th node in train, valid, test dataset respectively.
    '''
    def __init__(self, x: torch.Tensor, edge_index: torch.LongTensor, edge_weight: torch.Tensor,
                y: torch.Tensor, mask: torch.LongTensor, train_mask: torch.LongTensor, val_mask: torch.LongTensor, test_mask: torch.LongTensor):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_weight
        self.y = y
        self.num_classes = torch.unique(y).shape[0]
        self.num_nodes = x.shape[0]
        self.mask = mask
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.to_undirected()

    def get_split(self, split: str):
        tar_mask = {"train": self.train_mask, "valid": self.val_mask, "test": self.test_mask}[split]
        # print(tar_mask_id)
        return self.edge_index, self.edge_attr, tar_mask, self.y[tar_mask]

    def to_undirected(self):
        if not is_undirected(self.edge_index):
            self.edge_index, self.edge_attr = to_undirected(
                self.edge_index, self.edge_attr)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.y = self.y.to(device)
        self.mask = self.mask.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        return self

class dataset_heterophily(InMemoryDataset):
    def __init__(self,
                root='./data/',
                name=None,
                p2raw=None,
                train_percent=0.01,
                transform=None,
                pre_transform=None):

        existing_dataset = ['Chameleon', 'Film', 'Squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}'
            )
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(root, transform,
                                                pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def load_fixed_splits(root, name):
    """ loads saved fixed splits for dataset
    """
    split_pth = os.path.join(root, 'splits', f'{name}-splits.npy')
    print(split_pth)
    
    if not os.path.exists(split_pth):
        assert name in splits_drive_url.keys()
        gdown.download(
            id=splits_drive_url[name], \
            output=split_pth, quiet=False) 
    
    splits_lst = np.load(split_pth, allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst

def get_dataset(name, device, dir_path='./data/', transform=None):
    
    if name == 'Coauthor-CS':
        return Coauthor(root=dir_path, name='CS', transform=transform if transform is not None else T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=dir_path, name='Physics', transform=transform if transform is not None else T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=dir_path, name='Photo', transform=transform if transform is not None else T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=dir_path, name='Computers', transform=transform if transform is not None else T.NormalizeFeatures())

    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return Planetoid(dir_path, name, transform=transform if transform is not None else T.NormalizeFeatures())

    if name in ['WiKi-CS']:
        return WikiCS(dir_path, transform=transform if transform is not None else T.NormalizeFeatures())

    if name in ['Chameleon', 'Squirrel', 'Cornell', 'Texas']:
        pth = os.path.join(dir_path, f'{name}.pt')
        print(pth)
        if os.path.exists(pth):
            data = torch.load(pth)
            mask = data.mask
            split_idxs = load_fixed_splits(dir_path, name)[0]
            data.num_classes = data.y.unique().shape[0]
            train_mask, val_mask, test_mask = (
                split_idxs["train"],
                split_idxs["valid"],
                split_idxs["test"],
            )
            mask[train_mask] = 0
            mask[val_mask] = 1
            mask[test_mask] = 2
            data.mask = mask
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
            return [data]
        return dataset_heterophily(dir_path, name, transform=transform if transform is not None else T.NormalizeFeatures())
    
