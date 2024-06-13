import torch
import os.path as osp
import sys

from losses import InfoNCE
from base_model import CVA
import torch.nn.functional as F

from torch.optim import Adam
from eval import BaseEvaluator, LogReg, get_split
from contrast_model import DualBranchContrast
from datasets import get_dataset

from torch_geometric.nn import DenseGINConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import negative_sampling, to_undirected
import numpy as np
import argparse
import optuna
import math
import logging
import torch.nn as nn
import seaborn as sns
import time

import warnings
warnings.filterwarnings("ignore")

import faulthandler
faulthandler.enable()

def th_accuracy_score(pred, y):
    correct = (pred == y).sum().item()
    total = y.numel()
    accuracy = correct / total
    return accuracy


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogReg(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_test_acc = 0
        best_epoch = 0

        for epoch in range(self.num_epochs):
            classifier.train()
            optimizer.zero_grad()

            output = classifier(x[split['train']])
            loss = criterion(output_fn(output), y[split['train']])

            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.test_interval == 0:
                classifier.eval()
                y_pred = classifier(x[split['test']]).argmax(-1)
                y_test = y[split['test']]
                test_acc = th_accuracy_score(y_test, y_pred)

                test_micro = test_acc
                test_macro = 0 # f1_score(y_test, y_pred, average='macro')


                y_val = y[split['val']]
                y_pred = classifier(x[split['val']]).argmax(-1)
                
                val_micro = th_accuracy_score(y_val, y_pred)

                if val_micro > best_val_micro:
                    best_val_micro = val_micro
                    best_test_micro = test_micro
                    best_test_macro = test_macro
                    best_test_acc = test_acc
                    best_epoch = epoch


        return {
            'test_acc': best_test_acc,
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'val_acc': best_val_micro
        }

class DenseGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True, need_norm=False) -> None:
        super(DenseGCN, self).__init__()
        self.lin = Linear(input_dim, hidden_dim, bias=False, weight_initializer='glorot')
        self.need_norm = need_norm
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, adj, x):
        z = self.lin(x)
        if self.need_norm:
            d_inv_sqrt = torch.pow(torch.sum(adj, 1), -0.5)
            d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), torch.full_like(d_inv_sqrt, 0), d_inv_sqrt)
            d_inv_sqrt = torch.diag(d_inv_sqrt)
            adj = torch.mm(d_inv_sqrt, torch.mm(adj, d_inv_sqrt))
        out = torch.mm(adj, z)
        if self.bias is not None:
            return out + self.bias
        return out

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers, dropout=0.5, sparse=False, encoder_type='GCN'):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        # self.bns = nn.ModuleList()
        self.encoder_type = encoder_type
        self.norm = nn.BatchNorm1d(2*hidden_dim)

        if encoder_type == 'GCN':
            if num_layers == 1:
                self.layers.append(DenseGCN(input_dim, hidden_dim))
                # self.bns.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.layers.append(DenseGCN(input_dim, 2*hidden_dim, need_norm=False))
                # self.bns.append(nn.BatchNorm1d(2*hidden_dim))
                for _ in range(1, num_layers - 1):
                    self.layers.append(DenseGCN(2*hidden_dim, 2*hidden_dim, need_norm=False))
                    # self.bns.append(nn.BatchNorm1d(2*hidden_dim))
                self.layers.append(DenseGCN(2*hidden_dim, hidden_dim, need_norm=False))
                # self.bns.append(nn.BatchNorm1d(hidden_dim))

        elif encoder_type == 'GIN':
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            if num_layers == 1:
                net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(DenseGINConv(net))
                # self.bns.append(nn.BatchNorm1d(hidden_dim))
            else:
                net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(DenseGINConv(net))
                # self.bns.append(nn.BatchNorm1d(hidden_dim))
                for i in range(1, num_layers - 1):
                    net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                    self.layers.append(DenseGINConv(net))
                    # self.bns.append(nn.BatchNorm1d(hidden_dim))
                net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(DenseGINConv(net))
                # self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index=None, edge_weight=None, adj=None):
        z = x
        zs = []
        if edge_index is not None:
            for i, conv in enumerate(self.layers):
                z = conv(z, edge_index, edge_weight)
                z = self.activation(z)
                z = self.bns[i](z)
                zs.append(z)
        else:
            for i, conv in enumerate(self.layers):
                z = conv(adj=adj, x=z)
                z = z.squeeze(0) if self.encoder_type == 'GIN' else z
                # z = self.bns[i](z) # 
                if i != len(self.layers) - 1:
                    # z = self.bns[i](z)
                    z = self.norm(z)
                z = self.activation(z)
                # z = self.dropout(z)
        return z

class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim, proj_dim, tau=0.2, output_dim=0, norm_type='BN', per_epoch=True, sparse=False):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.per_epoch = per_epoch
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        if output_dim != 0:
            self.fc2 = torch.nn.Linear(proj_dim, output_dim)
            self.bias = nn.Parameter(torch.Tensor(output_dim), requires_grad=True)
        else:
            self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)
            self.bias = nn.Parameter(torch.Tensor(hidden_dim), requires_grad=True)
        self.norm_type = norm_type
        self.sparse = sparse
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.proj_dim)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x=None, ei=None, ew=None, adj=None):
        if ei is not None:
            z = self.encoder(x, ei, ew)
        else:
            z = self.encoder(adj=adj, x=x)
        return z

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z) + self.bias
    
    def loss(self, z1, z2, mean=True, batch_size=0):
        h1 = F.normalize(z1)  # [N, d]
        h2 = F.normalize(z2)

        h1_self_sim = torch.exp(torch.mm(h1, h1.t()) / self.tau)
        h2_self_sim = torch.exp(torch.mm(h2, h2.t()) / self.tau)
        comm_sim = torch.exp(torch.mm(h1, h2.t()) / self.tau)

        l1 = -torch.log(comm_sim.diag() / (h1_self_sim.sum(dim=1) - h1_self_sim.diag() + comm_sim.sum(dim=1))) 
        l2 = -torch.log(comm_sim.diag() / (h2_self_sim.sum(dim=1) - h2_self_sim.diag() + comm_sim.sum(dim=0))) 

        return (0.5 * (l1+l2)).mean() if mean else (0.5 * (l1+l2)).sum()

def search_hyper_params(trial : optuna.trial):
    global args, study_name

    if args.use_mask_ratio:
        if args.dataset == 'Cora':
            args.mask_ratio = trial.suggest_float("mask_ratio", 0.00, 0.2, step=0.05)
        elif args.dataset == 'Citeseer':
            args.mask_ratio = trial.suggest_float("mask_ratio", 0.00, 0.3, step=0.05)
        elif args.dataset == 'PubMed':
            args.mask_ratio = trial.suggest_float("mask_ratio", 0.1, 0.9, step=0.1)

    args.lr1 = trial.suggest_categorical("lr1", [5e-4, 1e-3, 5e-3]) # 5e-4
    args.lr2 = trial.suggest_categorical("lr2", [5e-4, 1e-3, 5e-3]) # 1e-3
    # wd1 = trial.suggest_categorical('wd1', [1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 0.0])
    # wd2 = trial.suggest_categorical('wd2', [1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 0.0])
    
    # hid_dim = trial.suggest_categorical("hid_dim", [128, 256])
    # proj_dim = trial.suggest_categorical("proj_dim",  [64, 128, 256])
    
    if args.dataset == 'Cora':
        # args.tau = trial.suggest_float("tau", 0.4, 0.65, step=0.05)
        # args.alpha = trial.suggest_float("alpha", 0.4, 0.65, step=0.05)
        # args.mask_ratio = trial.suggest_float("mask_ratio", 0.05, 0.2, step=0.05)
        # args.beta = trial.suggest_float("beta", 0.4, 0.65, step=0.05)
        args.tau = trial.suggest_float("tau", 0.1, 0.9, step=0.1)
        args.alpha = trial.suggest_float("alpha", 0.1, 0.9, step=0.1)
        args.mask_ratio = trial.suggest_float("mask_ratio", 0.00, 1.0, step=0.1)
        args.beta = trial.suggest_float("beta", 0.1, 0.9, step=0.1)


    elif args.dataset == 'CiteSeer':
        # args.tau = trial.suggest_float("tau", 0.7, 0.9, step=0.05)
        # args.alpha = trial.suggest_float("alpha", 0.1, 0.3, step=0.05)
        # args.mask_ratio = trial.suggest_float("mask_ratio", 0.00, 0.2, step=0.05)
        # args.beta = trial.suggest_float("beta", 0.1, 0.3, step=0.05)
        args.tau = trial.suggest_float("tau", 0.1, 0.9, step=0.1)
        args.alpha = trial.suggest_float("alpha", 0.1, 0.6, step=0.1)
        args.mask_ratio = trial.suggest_float("mask_ratio", 0.00, 0.8, step=0.1)
        args.beta = trial.suggest_float("beta", 0.1, 0.8, step=0.1)
    elif args.dataset == 'PubMed':
        # args.alpha = trial.suggest_float("alpha", 0.6, 0.9, step=0.1)
        # args.mask_ratio = trial.suggest_float("mask_ratio", 0.1, 0.9, step=0.1)
        # args.alpha = trial.suggest_float("alpha", 0.1, 5.0, step=0.1)
        args.alpha = trial.suggest_float("alpha", 1.0, 50.0, step=3.0)
        args.mask_ratio = trial.suggest_float("mask_ratio", 0.1, 0.9, step=0.1)

    elif args.dataset == 'Amazon-Computers':
        args.alpha = trial.suggest_float("alpha", 0.1, 0.9, step=0.1)
        args.mask_ratio = trial.suggest_float("mask_ratio", 0.1, 0.9, step=0.1)


    if args.dataset in ['Amazon-Photo']:
        args.mask_ratio = trial.suggest_float("mask_ratio", 0.1, 0.9, step=0.1)
        args.tau = trial.suggest_float("tau", 0.1, 0.9, step=0.1)
        # args.tau = trial.suggest_float("tau", 0.05, 0.2, step=0.05)
    elif args.dataset in ['Coauthor-CS', 'Coauthor-Phy', 'PubMed', 'Amazon-Computers']:
        args.tau = trial.suggest_float("tau", 0.1, 0.9, step=0.1)
    elif args.dataset in ['WiKi-CS']:
        args.tau = trial.suggest_float("tau", 0.1, 0.9, step=0.1)
    
    if args.encoder_type == 'GCN':
        args.num_layers = trial.suggest_categorical("num_layers", [2])
    elif args.encoder_type == 'GIN':
        args.num_layers = trial.suggest_categorical("num_layers", [3, 4, 5])

    # args.wd1 = wd1
    # args.wd2 = wd2
    # args.hid_dim = hid_dim
    # args.proj_dim = proj_dim

    # args.alpha = trial.suggest_categorical("alpha", [ 1,4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.9, 1.95, 2.0, 2.05, 2.1]) #2, 5, 8, 10, 15, 20, 50])
    if args.dataset == 'Amazon-Photo':
        args.alpha = trial.suggest_categorical("alpha", [0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9, 2.0])
    elif args.dataset in ['Coauthor-Phy']:
        args.alpha = trial.suggest_categorical("alpha", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 5.0, 10.0, 15.0, 20.0])
    elif args.dataset == 'Coauthor-CS':
        args.alpha = trial.suggest_float("alpha", 4.7, 5.3, step=0.1)
    elif args.dataset == 'WiKi-CS':
        args.alpha = trial.suggest_float("alpha", 0.1, 0.9, step=0.1)
        args.mask_ratio = trial.suggest_float("mask_ratio", 0.1, 0.9, step=0.1)

    # args.alpha = trial.suggest_categorical("alpha", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) # Amazon-Computers

    if args.dataset == 'Amazon-Photo':
        args.beta = trial.suggest_categorical("beta", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    elif args.dataset in ['Coauthor-Phy', 'Amazon-Computers', 'WiKi-CS']:
        args.beta = trial.suggest_float("beta", 0.0 , 0.9, step=0.1)
    elif args.dataset == 'Coauthor-CS':
        args.beta = trial.suggest_float("beta", 0.4 , 0.7, step=0.1)
    elif args.dataset in ['PubMed']:
        # args.beta = trial.suggest_float("beta", 0.0, 5.0, step = 0.1)
        args.beta = trial.suggest_float("beta", 1.0, 10.0, step=1.0)

    return ULA(args=args, trial_id=trial.number)


def sim(z1, z2, method='cos'):

    z1 = z1.norm(dim=1, p=2, keepdim=True)
    z2 = z2.norm(dim=1, p=2, keepdim=True)
    if method == 'cos':
        return 1.0 - (1.0 + torch.mm(z1.T, z2))/2.0  
    elif method == 'mse':
        return 1.0 - F.mse_loss(z1, z2)
    elif method == 'exp':
        return torch.exp(1.0 - F.l1_loss(z1, z2))

def ULA(args, trial_id=0):
    global data, U, Lamb, C, device, study_name


    logging.info('\n\nstart Training...')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.info(f'alpha : {args.alpha:.5f}, beta : {args.beta:.5f}, lr1 : {args.lr1:.5f}, lr2 : {args.lr2:.5f}, tau: {args.tau:.3f}, use_mask_ratio: {args.use_mask_ratio}, mask_ratio : {args.mask_ratio}, num_layers: {args.num_layers}, sim_method: {args.sim_method}, botune: {args.botune}, pe: {args.pe}')
    data.edge_neg_index = negative_sampling(data.edge_index, num_nodes=data.x.shape[0])
    data = data.to(device)
    tmp_U = torch.from_numpy(U).to(device).to(torch.float32)
    tmp_U.requires_grad=False

    tmp_Lamb = np.diag(Lamb)
    tmp_C = torch.from_numpy(C).to(device).to(torch.float32)
    tmp_C.requires_grad=False

    num_node = data.x.size(0)
    num_feat = data.x.size(1)
    num_class = data.y.max().item() + 1

    if args.use_contrast_mode:
        contrast_model = DualBranchContrast(loss=InfoNCE(tau=args.tau), mode='L2L', intraview_negs=False).to(device)

    x2 = torch.mm(tmp_U, tmp_C)
    adj2 =  torch.eye(num_node).to(device) - torch.mm(tmp_U, torch.mm(torch.from_numpy(tmp_Lamb).to(torch.float32).to(device), tmp_U.T))

    best_epoch = 0
    
    torch.cuda.empty_cache()


    gconv = GConv(input_dim=num_feat, hidden_dim=args.hid_dim, activation=torch.nn.ReLU, num_layers=args.num_layers, encoder_type=args.encoder_type).to(device)
    encoder_model = Encoder(encoder=gconv, hidden_dim=args.hid_dim, proj_dim=args.proj_dim, tau=args.tau).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    gconv_ib = GConv(input_dim=num_feat, hidden_dim=args.hid_dim, activation=torch.nn.ReLU, num_layers=args.num_layers, encoder_type=args.encoder_type).to(device)
    encoder_model_ib = Encoder(encoder=gconv_ib, hidden_dim=args.hid_dim, proj_dim=args.proj_dim, tau=args.tau).to(device)
    optimizer_ib = Adam(encoder_model_ib.parameters(), lr=args.lr1, weight_decay=args.wd1)

    generator = CVA(num_class, num_node, num_feat, device, use_conv=args.use_conv, Lamb=tmp_Lamb, ratio=args.mask_ratio, C=tmp_C, sparse=False, pe=args.pe).to(device)
    optimizer_gene = Adam(generator.parameters(), lr=args.lr2, weight_decay=args.wd2)

    best_test_acc = 0.0

    for epoch in range(1, args.num_epochs+1):
        if args.use_contrast_mode:
            train_epoch(encoder_model, encoder_model_ib, generator, optimizer, optimizer_ib, optimizer_gene, contrast_model, tmp_U, adj2, x2, args)
        else:
            train_epoch(encoder_model, encoder_model_ib, generator, optimizer, optimizer_ib, optimizer_gene, None, tmp_U, adj2, x2, args)

        result = {}
        if epoch % args.eval_intervals == 0:

            encoder_model.eval()
            encoder_model_ib.eval()
            generator.eval()
            with torch.no_grad():
                z = encoder_model(x=x2, adj=adj2) 
                x1, adj1 = generator(U=tmp_U, x=x2) 
                z_ = encoder_model(x=x1, adj=adj1)
                z = (z + z_) / 2.0
            
            best_result = {
                'test_acc': 0.,
                'micro_f1': 0.,
                'macro_f1': 0.,
                'val_acc': 0.
            }

            for decay in [0.0, 0.001, 0.005, 0.01, 0.1]:
                split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
                result = LREvaluator(weight_decay=decay)(z, data.y, split)
                if result['val_acc'] > best_result['val_acc']:
                    best_result = result
            
            test_acc = best_result['test_acc']
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
        torch.cuda.empty_cache()

    print(f"best_epoch={best_epoch}, best_test_acc={best_test_acc}")
    return best_test_acc

def train_epoch(encoder_model, encoder_model_ib, generator, optimizer, optimizer_ib, optimizer_gene, contrast_model, tmp_U, adj2, x2, args):
    encoder_model.train()
    generator.eval()
    optimizer.zero_grad()
    x1, adj1 = generator(U=tmp_U, x=x2)
    z1, z2 = [encoder_model.project(x) for x in [encoder_model(x=x1.detach(), adj=adj1.detach()), encoder_model(x=x2.detach(), adj=adj2.detach())]]
    _v1, _v2 = [encoder_model_ib.project(x) for x in [encoder_model_ib(x=x1.detach(), adj=adj1.detach()), encoder_model_ib(x=x2.detach(), adj=adj2.detach())]]
    torch.cuda.empty_cache()
    if contrast_model is not None:
        infomax_loss = contrast_model(z1, z2)
    else:
        infomax_loss = encoder_model.loss(z1, z2)
    
    encoder_loss = infomax_loss 
    encoder_loss.backward()
    optimizer.step()

    encoder_model.eval()
    encoder_model_ib.train()
    if contrast_model is not None:
        bn_loss = contrast_model(_v1, z1.detach()) + contrast_model(_v2, z2.detach())
    else:
        bn_loss = encoder_model.loss(_v1, z1.detach()) + encoder_model.loss(_v2, z2.detach())
    bn_loss = args.beta * bn_loss
    bn_loss.backward()
    optimizer_ib.step()

    encoder_model_ib.eval()
    generator.train()
    optimizer_gene.zero_grad()
    v1_, v2_ = [encoder_model_ib.project(x) for x in [encoder_model_ib(x=x1.detach(), adj=adj1.detach()), encoder_model_ib(x=x2.detach(), adj=adj2.detach())]]
    if args.sim_method == 'none':
        if contrast_model is not None:
            infomin_loss = contrast_model(v1_, v2_)
        else:
            infomin_loss = encoder_model.loss(v1_, v2_)
    else:
        infomin_loss = sim(v1_, v2_, method=args.sim_method)
    generator_loss = -1.0 * infomin_loss
    if args.feat_strategy not in ['learnable', 'SimCFA', 'rawX']:
        if args.topo_strategy not in ['learnable', 'SimMTA']:
            reg_loss = 0
            generator_loss = generator_loss + reg_loss
        else:
            reg_loss = args.alpha * (generator.Lamb_JSD_loss())
            generator_loss = generator_loss + reg_loss
    elif args.feat_strategy == 'rawX':
        if args.topo_strategy not in ['learnable', 'SimMTA']:
            reg_loss = args.alpha * (generator.X_JSD_Loss(tmp_U, x1))
            generator_loss = generator_loss + reg_loss
        else:
            reg_loss = args.alpha * (generator.X_JSD_Loss(tmp_U, x1) + generator.Lamb_JSD_loss())
            generator_loss = generator_loss + reg_loss
    else:
        if args.topo_strategy not in ['learnable', 'SimMTA']: # none, msking 
            reg_loss = args.beta * (generator.C_JSD_Loss())
            generator_loss = generator_loss + reg_loss
        else:
            reg_loss = args.alpha * (generator.C_JSD_Loss() + generator.Lamb_JSD_loss())
            generator_loss = generator_loss + reg_loss
    if torch.isnan(generator_loss):
        print("loss is nan")
    generator_loss.backward()
    optimizer_gene.step()

    return infomax_loss.item(), reg_loss.item(), encoder_loss.item(), infomin_loss.item(), bn_loss.item(), generator_loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--max_iter', type=int, default=1500)

    parser.add_argument('--train_ratio', type=float, default=0.1, help='train_ratio of the data')
    parser.add_argument('--test_ratio', type=float, default=0.8, help='test_ratio of the data')

    parser.add_argument('--per_epoch', action='store_true')  
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--encoder_type', type=str, default='GCN', choices=['GCN', 'GIN', 'GCN-Res', 'GIN-Res', 'GraphSAGE'])

    parser.add_argument('--lr1', type=float, default=1e-4)
    parser.add_argument('--lr2', type=float, default=1e-4)
    parser.add_argument('--wd1', type= float, default=1e-5)
    parser.add_argument('--wd2', type= float, default=1e-5)
    parser.add_argument('--optruns', type=int, default=100)
    parser.add_argument('--runs', type=int, default=1, help='use in test stage with 5 different data indicies permutations')
    parser.add_argument('--early_stop', action='store_false')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--use_conv', action='store_false')
    parser.add_argument('--topo_strategy', type=str, default='SimMTA', choices=['masking', 'learnable', 'SimMTA', 'none'])
    parser.add_argument('--feat_strategy', type=str, default="SimCFA", choices=["learnable", "masking", "COSTA", "SimCFA", "rawX", "none"])
    parser.add_argument('--feat_mask_ratio', type=float, default=0.2)
    parser.add_argument('--use_mask_ratio',action='store_true')

    parser.add_argument('--sim_method', type=str, default='cos', choices=['none', 'cos', 'exp', 'mse'])
    parser.add_argument('--botune', action='store_true')
    parser.add_argument('--eval_intervals', type=int, default=10)

    parser.add_argument('--mask_ratio', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--base_memory', type=float, default=0.0)

    parser.add_argument('--load_pth', action='store_true')
    parser.add_argument('--model_ib_pth', type=str, default='')
    parser.add_argument('--generator_pth', type=str, default='')
    parser.add_argument('--encoder_pth', type=str, default='')
    parser.add_argument('--pe', type=str, default='Sine')
    parser.add_argument('--COSTA_ratio', type=float, default=1.0)
    parser.add_argument('--mode', type=str, default='l2l', choices=['l2l', 'l2g', 'g2l', 'g2g'])
    parser.add_argument('--use_contrast_mode', action='store_true')

    args = parser.parse_args()

    torch.cuda.set_device(int(args.device.split(':')[-1]))
    device = torch.device(args.device)

    path = '~/code/data/' 
    path = osp.join(path, args.dataset) 
    dataset = get_dataset(args.dataset, device=device, dir_path=path)
    data = dataset[0].to(device)
    if args.dataset == 'WiKi-CS':
        std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
        data.x = (data.x - mean) / std
        data.edge_index = to_undirected(data.edge_index)

    U, Lamb, C = None, None, None
    if osp.exists(f'~/code/LOC/data/npy/{args.dataset}_sorted_eigvecs.npy'):
        U = np.load(f"~/code/data/npy/{args.dataset}_sorted_eigvecs.npy")
    if osp.exists(f'~/code/data/npy/{args.dataset}_sorted_eigvals.npy'):
        Lamb = np.load(f"~/code/data/npy/{args.dataset}_sorted_eigvals.npy")
    if osp.exists(f'~/code/data/npy/{args.dataset}_singal_amps.npy'):
        C = np.load(f"~/code/data/npy/{args.dataset}_singal_amps.npy")

    study_name = 'avg_LOC_'+f'{args.dataset}'
    if args.use_mask_ratio:
        study_name += '_masked'

    if args.use_contrast_mode:
        study_name += '_contrast'

    study_name += f'_encoder_{args.encoder_type}'
    study_name += f'_layer_{args.num_layers}'

    if args.sim_method != 'none':
        study_name += '_'+args.sim_method

    if args.feat_strategy == 'masking':
        study_name += '_FtMsk'
    elif args.feat_strategy == 'none':
        study_name += '_NoFeat'
    elif args.feat_strategy == 'learnable':
        if args.use_conv:
            study_name += '_FIA_conv'
        else:
            study_name += '_FIA_MLP'
    elif args.feat_strategy == 'COSTA':
        study_name += '_COSTA'
    elif args.feat_strategy == 'SimCFA':
        study_name += '_SimCFA'

    if args.topo_strategy == 'masking':
        study_name += '_TpMsk'
    elif args.topo_strategy == 'none':
        study_name += '_NoTp'
    elif args.topo_strategy == 'learnable':
        study_name += '_Specformer'
    elif args.topo_strategy == 'SimMTA':
        study_name += '_SimMTA'

    study_name += '_res'
    
    study_name += '_' + args.pe
    study_name += '_' + str(args.COSTA_ratio)

    import pathlib
    pathlib.Path(
        f'~/code/pts/{args.dataset}/{study_name}/gene/'
    ).mkdir(parents=True, exist_ok=True)

    pathlib.Path(
        f'~/code/pts/{args.dataset}/{study_name}/encoder/'
    ).mkdir(parents=True, exist_ok=True)

    pathlib.Path(
        f'~/code/bo_dbs/{args.dataset}/'
    ).mkdir(parents=True, exist_ok=True)

    pathlib.Path(
        f'~/code/pics/{args.dataset}/{study_name}/2D/'
    ).mkdir(parents=True, exist_ok=True)

    pathlib.Path(
        f'~/code/pics/{args.dataset}/{study_name}/3D/'
    ).mkdir(parents=True, exist_ok=True)

    if args.botune:
        study = optuna.create_study(direction="maximize",
                                    storage='sqlite:///' + f'~/code/bo_dbs/{args.dataset}/' + study_name+ '.db',
                                    study_name=study_name,
                                    load_if_exists=True)

        study.optimize(search_hyper_params, n_trials=args.optruns)

        print("best params", study.best_params)
        print("best val_acc", study.best_value)
    else: 
        ULA(args=args)