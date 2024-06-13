from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS
import torch_geometric.transforms as T

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