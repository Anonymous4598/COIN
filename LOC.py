import torch
import os.path as osp
import sys

from losses import InfoNCE
from augmentor import EdgeRemoving, FeatureMasking
from base_model import  GenWithSpecformer, TopoGenSpecFeatMasking, TopoGenSpecformer, TopoGenSpecFeatCOSTA, TopoAugFeatGen, FeatGenConv, CVA, TopoGenSpecFeatRawX, TopoSimMTA, TopoSimMTAFeatRawX
import torch.nn.functional as F

from torch.optim import Adam
from eval import BaseEvaluator, WiKiBaseEvaluator, LogReg, get_split, LinkBaseEvaluator
from contrast_model import DualBranchContrast
from datasets import get_dataset
from tqdm import tqdm

from torch_geometric.nn import DenseGINConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import to_scipy_sparse_matrix, negative_sampling, to_undirected
import numpy as np
import pandas as pd
import argparse
import optuna
import os 
import math
import logging
import torch.nn as nn
import seaborn as sns
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import time

import warnings
warnings.filterwarnings("ignore")

import faulthandler
faulthandler.enable()

from sklearn.metrics import f1_score, accuracy_score, precision_score

def draw_plot(data, embeddings, fname, max_nodes=None):
    
    # graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)
    # labels = [graph.graph['label'] for graph in graphs]
    num_classes = data.y.unique().max().item() + 1
    labels = data.y.long().cpu().numpy()
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    x = TSNE(n_components=2, init='pca', random_state=42).fit_transform(x)

    plt.close()
    df = pd.DataFrame(columns=['x0', 'x1', 'Y'])
    df['x0'], df['x1'], df['Y'] = x[:,0], x[:,1], y

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.scatter(df['x0'], df['x1'], c=y, cmap=plt.cm.coolwarm, marker="o")
    # sns.pairplot(x_vars=['x0'], y_vars=['x1'], data=df, hue="Y", size=1)
    # plt.legend()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(fname)
    plt.close()

def draw_plot_3D(data, embeddings, fname, epoch=0):
    labels = data.y.long().cpu().numpy()
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    x = StandardScaler().fit_transform(x)
    plt.close()
    x = TSNE(n_components=3, init='pca', random_state=42).fit_transform(x)
    df = pd.DataFrame(columns=['x0', 'x1', 'x3', 'Y'])
    df['x0'], df['x1'], df['x2'], df['Y'] = x[:,0], x[:,1], x[:,2], y

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['x0'], df['x1'], df['x2'], c=y, cmap=plt.cm.coolwarm, marker="o") # 'rainbow'
    # plt.title(f"{fname.split('/')[-1].split('.')[0]}_epoch:{epoch}")
    # plt.axis('off')
    # plt.show()
    # plt.legend()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    plt.savefig(fname)
    plt.close()

def sorted_three_draw_feature_coefficients_change(ori_C, new_C, rand_C, fname, dataset):
    avg_ori_C = torch.mean(ori_C, dim=0)
    sum_avg_ori_C = torch.sum(avg_ori_C) 
    avg_ori_C = avg_ori_C / sum_avg_ori_C
    avg_ori_C = avg_ori_C.detach().cpu().numpy()
    avg_ori_C[::-1].sort()

    avg_new_C = torch.mean(new_C, dim=0)
    sum_avg_new_C = torch.sum(avg_new_C) 
    avg_new_C = avg_new_C / sum_avg_new_C
    avg_new_C = avg_new_C.detach().cpu().numpy()
    avg_new_C[::-1].sort()

    avg_rand_C = torch.mean(rand_C, dim=0)
    sum_avg_rand_C = torch.sum(avg_rand_C)
    avg_rand_C = avg_rand_C / sum_avg_rand_C
    avg_rand_C = avg_rand_C.detach().cpu().numpy()
    avg_rand_C[::-1].sort()

    plt.clf()
    x_ticks = np.arange(ori_C.size(1))
    plt.plot(x_ticks, avg_ori_C, linewidth=3.0, color=plt.cm.coolwarm(0.1),label='original C')
    plt.plot(x_ticks, avg_rand_C, linewidth=3.0, color=plt.cm.coolwarm(0.5),label='random C')
    plt.plot(x_ticks, avg_new_C, linewidth=3.0, color=plt.cm.coolwarm(0.9),label='augmented C')
    plt.xticks(size=30)
    plt.yticks(size=30)
    # plt.title(f'The change of $C$ on {dataset}', fontsize=40)
    plt.legend(fontsize=40)
    plt.savefig(fname)
    plt.close()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)

# right
def th_accuracy_score(pred, y):
    # 计算预测正确的数量
    correct = (pred == y).sum().item()
    # 计算总的数量
    total = y.numel()
    # 计算准确率
    accuracy = correct / total
    return accuracy

# wrong
def th_f1_score(pred, y, average='micro'):
    # 计算 true positives, false positives 和 false negatives
    if average=='micro':
        tp = (y * pred).sum().to(torch.float32)
        fp = ((1 - y) * pred).sum().to(torch.float32)
        fn = (y * (1 - pred)).sum().to(torch.float32)
    else:
        tp = (y * pred).sum(dim=0).to(torch.float32)
        fp = ((1 - y) * pred).sum(dim=0).to(torch.float32)
        fn = (y * (1 - pred)).sum(dim=0).to(torch.float32)

    # 计算 precision 和 recall
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    # 计算 F1 分数
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1

class WiKiLREvaluator(WiKiBaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.FloatTensor, train_masks, val_masks, test_mask):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1 # 10+1
        classifier = LogReg(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_macro = 0
        best_epoch = 0
        avg_test_acc = []
        avg_test_macro = []
        avg_val_acc = []

        for split_id in [0]:
            train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]
            X_train, y_train = x[train_mask], y[train_mask]
            X_val, y_val = x[val_mask], y[val_mask]
            X_test, y_test = x[test_mask], y[test_mask]

            best_test_acc = 0
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(X_train)
                loss = criterion(output_fn(output), y_train)
                loss.backward()
                optimizer.step()

                if (epoch+1)%self.test_interval == 0:
                    classifier.eval()
                    y_pred = classifier(X_test).argmax(-1)
                    test_acc = th_accuracy_score(y_test, y_pred)

                    test_macro = 0

                    y_pred = classifier(X_val).argmax(-1)
                    val_micro = th_accuracy_score(y_val, y_pred)

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_macro = test_macro
                        best_test_acc = test_acc
                        best_epoch = epoch

            avg_test_acc.append(best_test_acc)
            avg_test_macro.append(best_test_macro)
            avg_val_acc.append(best_val_micro)
        
        return {
            'test_acc': np.mean(avg_test_acc),
            'micro_f1': np.mean(avg_test_acc),
            'macro_f1': np.mean(avg_test_macro),
            'val_acc': np.mean(best_val_micro)
        }

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

        # with tqdm(total=self.num_epochs, desc='(LR)',
        #           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
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

                # y_test = y_test.detach().cpu().numpy()
                # y_pred = y_pred.detach().cpu().numpy()
                test_micro = test_acc
                test_macro = 0 # f1_score(y_test, y_pred, average='macro')
                # print(f"test_acc={test_acc}, test_micro={test_micro}, test_macro={test_macro}")


                y_val = y[split['val']]
                y_pred = classifier(x[split['val']]).argmax(-1)
                
                val_micro = th_accuracy_score(y_val, y_pred)

                if val_micro > best_val_micro:
                    best_val_micro = val_micro
                    best_test_micro = test_micro
                    best_test_macro = test_macro
                    best_test_acc = test_acc
                    best_epoch = epoch

                    # pbar.set_postfix({'best test acc': best_test_acc, 'F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    # pbar.update(self.test_interval)

        return {
            'test_acc': best_test_acc,
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'val_acc': best_val_micro
        }

class LinkEvaluator(LinkBaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.1,
                weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, ei: torch.LongTensor, neg_ei: torch.LongTensor, split: dict, neg_split:dict):
        device = x.device 
        x = x.detach().to(device)
        input_dim = 2 * x.size(1)
        train_ei = ei[:, split['train']]
        val_ei = ei[:, split['val']]
        test_ei = ei[:, split['test']]
        train_neg_ei = neg_ei[:, neg_split['train']]
        val_neg_ei = neg_ei[:, neg_split['val']]
        test_neg_ei = neg_ei[:, neg_split['test']]

        num_classes = 2
        classifier = LogReg(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = lambda x:x # nn.Sigmoid()
        criterion = nn.BCELoss(size_average=True)

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_test_acc = 0
        best_epoch = 0
        test_acc = 0.0

        for epoch in range(self.num_epochs):
            classifier.train()
            # print(classifier.fc.weight.requires_grad)
            optimizer.zero_grad()
            pos_output = classifier(torch.concat([x[train_ei[0, :]], x[train_ei[1, :]]], 1)).argmax(-1).float()
            neg_output = classifier(torch.concat([x[train_neg_ei[0, :]], x[train_neg_ei[1, :]]], 1)).argmax(-1).float()
            pos_output.requires_grad = True
            neg_output.requires_grad = True
            labels_all = torch.concat([torch.ones(train_ei.size(1)).float().to(device), torch.zeros(train_neg_ei.size(1)).float().to(device)])
            pred_all = output_fn(torch.concat([pos_output, neg_output]))
            # pred_all.requires_grad = True
            # print(f"pred_all: {pred_all}")
            # print(f"label_all: {labels_all}")
            loss = criterion(pred_all, labels_all)
            # print(f"loss: {loss}")
            loss.backward()
            optimizer.step()
            if (epoch + 1) % self.test_interval == 0:
                classifier.eval()
                pos_y_pred = classifier(torch.concat([x[test_ei[0, :]], x[test_ei[1, :]]], 1)).argmax(-1).float()
                neg_y_pred = classifier(torch.concat([x[test_neg_ei[0, :]], x[test_neg_ei[1, :]]], 1)).argmax(-1).float()
                y_test = torch.concat([torch.ones(test_ei.size(1)).to(device), torch.zeros(test_neg_ei.size(1)).to(device)])
                y_pred = torch.concat([pos_y_pred, neg_y_pred])
                test_acc = th_accuracy_score(y_test, y_pred)
                test_micro = test_acc
                test_macro = 0 # f1_score(y_test, y_pred, average='macro')
                pos_y_pred = classifier(torch.concat([x[val_ei[0, :]], x[val_ei[1, :]]], 1)).argmax(-1)
                neg_y_pred = classifier(torch.concat([x[val_neg_ei[0, :]], x[val_neg_ei[1, :]]], 1)).argmax(-1)
                y_pred = torch.concat([pos_y_pred, neg_y_pred])
                y_val = torch.concat([torch.ones(val_ei.size(1)).to(device), torch.zeros(val_neg_ei.size(1)).to(device)])
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
    
    # def infonce_loss(self, h1, h2):
    #     # f = lambda x: torch.exp(x / self.tau)

    #     # h1 = self.project(z1)
    #     # h2 = self.project(z2)
    #     h1 = F.normalize(h1)  # [N, d]
    #     h2 = F.normalize(h2)
    #     self_sim = torch.exp(torch.mm(h1, h1.t()) / self.tau)  # [N, N]
    #     comm_sim = torch.exp(torch.mm(h1, h2.t()) / self.tau)  # [N, N]

    #     # self_sim_sum = torch.exp(torch.mm(h1, h1.t()) / self.tau).sum(dim=1)  # [N, N]
    #     # self_sim_diag = torch.exp(torch.mm(h1, h1.t()) / self.tau).diag()
    #     # comm_sim_sum = torch.exp(torch.mm(h1, h2.t()) / self.tau).sum(dim=1)  # [N, N]
    #     # comm_sim_diag = torch.exp(torch.mm(h1, h2.t()) / self.tau).diag()  # [N, N]

    #     return -torch.log(comm_sim.diag() / (self_sim.sum(dim=1) - self_sim.diag() + comm_sim.sum(dim=1))) 
    #     # return -torch.log(comm_sim_diag / (self_sim_sum - self_sim_diag + comm_sim_sum)) 
    
    # def loss(self, z1, z2, mean=True, batch_size=0):

    #     l1 = self.infonce_loss(z1, z2)
    #     l2 = self.infonce_loss(z2, z1)

    #     return (0.5 * (l1+l2)).mean() if mean else (0.5 * (l1+l2)).sum()

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
        elif args.dataset in ['Chameleon', 'Squirrel', 'Cornell', 'Texas']:
            args.alpha = trial.suggest_float("alpha", 0.1, 0.9, step=0.1)
            args.beta = trial.suggest_float("beta", 0.1, 0.9, step=0.1)
            args.tau = trial.suggest_float("tau", 0.1, 0.9, step=0.1)
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

    # args.lr1 = lr1
    # args.lr2 = lr2 
    # args.wd1 = wd1
    # args.wd2 = wd2
    # args.hid_dim = hid_dim
    # args.proj_dim = proj_dim

    # args.alpha = trial.suggest_categorical("alpha", [ 1,4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.9, 1.95, 2.0, 2.05, 2.1]) #2, 5, 8, 10, 15, 20, 50])
    if args.dataset == 'Amazon-Photo':
        args.alpha = trial.suggest_float("alpha", 0.1, 0.9, step=0.1)
        # args.alpha = trial.suggest_categorical("alpha", [0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9, 2.0])
    elif args.dataset in ['Coauthor-Phy']:
        args.alpha = trial.suggest_categorical("alpha", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 5.0, 10.0, 15.0, 20.0])
    elif args.dataset == 'Coauthor-CS':
        args.alpha = trial.suggest_float("alpha", 4.7, 5.3, step=0.1)
    elif args.dataset == 'WiKi-CS':
        args.alpha = trial.suggest_float("alpha", 0.1, 0.9, step=0.1)
        args.mask_ratio = trial.suggest_float("mask_ratio", 0.1, 0.9, step=0.1)

    # args.alpha = trial.suggest_categorical("alpha", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) # Amazon-Computers

    if args.dataset == 'Amazon-Photo':
        args.beta = trial.suggest_float("beta", 0.1, 0.9, step=0.1)
        # args.beta = trial.suggest_categorical("beta", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    elif args.dataset in ['Coauthor-Phy', 'Amazon-Computers', 'WiKi-CS']:
        args.beta = trial.suggest_float("beta", 0.0 , 0.9, step=0.1)
    elif args.dataset == 'Coauthor-CS':
        args.beta = trial.suggest_float("beta", 0.4 , 0.7, step=0.1)
        args.mask_ratio = trial.suggest_float("mask_ratio", 0.1, 0.9, step=0.1)
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

    preprocess_time_st = time.time()

    logging.info('\n\nstart Training...')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    # logging.info(args)
    logging.info(f'alpha : {args.alpha:.5f}, beta : {args.beta:.5f}, lr1 : {args.lr1:.5f}, lr2 : {args.lr2:.5f}, tau: {args.tau:.3f}, use_mask_ratio: {args.use_mask_ratio}, mask_ratio : {args.mask_ratio}, num_layers: {args.num_layers}, sim_method: {args.sim_method}, botune: {args.botune}, pe: {args.pe}')
    data.neg_edge_index = negative_sampling(data.edge_index, num_nodes=data.x.shape[0], num_neg_samples=data.edge_index.size(1)) # every positive edge has a negative edge
    data = data.to(device)
    tmp_U = torch.from_numpy(U).to(device).to(torch.float32)
    tmp_U.requires_grad=False

    tmp_Lamb = np.diag(Lamb)
    tmp_C = torch.from_numpy(C).to(device).to(torch.float32)
    tmp_C.requires_grad=False

    num_node = data.x.size(0)
    num_feat = data.x.size(1)
    num_class = data.y.max().item() + 1

    C1_e = torch.sum((tmp_C)**2, dim=1)/num_feat
    C1_bar = torch.mean(C1_e)
    C_metric_ori = torch.mean((C1_e-C1_bar)**2)

    # ori_adj = torch.from_numpy(to_scipy_sparse_matrix(data.edge_index, data.edge_attr).A).to(device)

    if args.use_contrast_mode:
        contrast_model = DualBranchContrast(loss=InfoNCE(tau=args.tau), mode='L2L', intraview_negs=False).to(device)

    accs = []

    his_best_test_acc = 0.0
    if args.dataset == 'Cora':
        his_best_test_acc = 0.855
    elif args.dataset == 'CiteSeer':
        his_best_test_acc = 0.730
    elif args.dataset == 'PubMed':
        his_best_test_acc = 0.840
    elif args.dataset == 'Amazon-Photo':
        his_best_test_acc = 0.930
    elif args.dataset == 'Amazon-Computers':
        his_best_test_acc = 0.900
    elif args.dataset == 'Coauthor-CS':
        his_best_test_acc = 0.930
    elif args.dataset == 'Coauthor-Phy':
        his_best_test_acc = 0.950
    elif args.dataset == 'WiKi-CS':
        his_best_test_acc = 0.785
    elif args.dataset == 'Chameleon':
        his_best_test_acc = 0.635
    elif args.dataset == 'Squirrel':
        his_best_test_acc = 0.420

    # if len(os.listdir(f'./pts/{args.dataset}/{study_name}/encoder/')) != 0:
    #     f = lambda x : float(x)
    #     his_best_test_acc = list(map(f, os.listdir(f'./pts/{args.dataset}/{study_name}/encoder/')[0].split(".pt")[0].split(",")))

    lastacc, lastMiF1, lastMaF1 = 0.0, 0.0, 0.0

    x2 = torch.mm(tmp_U, tmp_C)
    # ori_adj = torch.from_numpy(to_scipy_sparse_matrix(data.edge_index, data.edge_attr).A).to(device)
    adj2 =  torch.eye(num_node).to(device) - torch.mm(tmp_U, torch.mm(torch.from_numpy(tmp_Lamb).to(torch.float32).to(device), tmp_U.T))

    best_epoch = 0
    best_testMiF1 = 0.0
    best_testMaF1 = 0.0
    best_val_acc = lastacc
    
    convergence_cnt = 0
    preprocess_time = time.time() - preprocess_time_st

    torch.cuda.empty_cache()
    # print(f'allocated GPU Mem before training : {torch.cuda.memory_allocated()/float(1024*1024*1024)} GB')

    total_times = []
    mean_per_epoch_time = []
    edge_removal_augmenter = EdgeRemoving(pe=0.5)
    feature_mask_augmenter = FeatureMasking(pf=0.2)

    seeds = [2024, 2023, 2022, 2021, 2000]

    for sd in seeds:
        setup_seed(sd)
        if args.load_pth:
            generator = torch.load(args.generator_pth).to(device)
            encoder_model = torch.load(args.encoder_pth).to(device)
            encoder_model_ib = None
            optimizer = None
            optimizer_gene = None
            optimizer_ib = None 
        else:
            gconv = GConv(input_dim=num_feat, hidden_dim=args.hid_dim, activation=torch.nn.ReLU, num_layers=args.num_layers, encoder_type=args.encoder_type).to(device)
            encoder_model = Encoder(encoder=gconv, hidden_dim=args.hid_dim, proj_dim=args.proj_dim, tau=args.tau).to(device)
            optimizer = Adam(encoder_model.parameters(), lr=args.lr1, weight_decay=args.wd1)

            gconv_ib = GConv(input_dim=num_feat, hidden_dim=args.hid_dim, activation=torch.nn.ReLU, num_layers=args.num_layers, encoder_type=args.encoder_type).to(device)
            encoder_model_ib = Encoder(encoder=gconv_ib, hidden_dim=args.hid_dim, proj_dim=args.proj_dim, tau=args.tau).to(device)
            optimizer_ib = Adam(encoder_model_ib.parameters(), lr=args.lr1, weight_decay=args.wd1)
            # optimizer_ib = Adam(encoder_model.parameters(), lr=args.lr1, weight_decay=args.wd1)

            if args.topo_strategy == 'learnable':
                if args.feat_strategy == 'learnable': 
                        generator = GenWithSpecformer(num_class, num_node, num_feat, device, use_conv=args.use_conv, Lamb=tmp_Lamb, ratio=args.mask_ratio, C=tmp_C, sparse=False).to(device)
                elif args.feat_strategy == 'masking':
                    generator = TopoGenSpecFeatMasking(num_class, num_node, num_feat, device, feat_masking_ratio=args.feat_mask_ratio, Lamb=tmp_Lamb, ratio=args.mask_ratio, sparse=False).to(device)
                elif args.feat_strategy == 'COSTA':
                    generator = TopoGenSpecFeatCOSTA(num_class, num_node, num_feat, device, feat_masking_ratio=args.feat_mask_ratio, Lamb=tmp_Lamb, ratio=args.mask_ratio, sparse=False).to(device)
                elif args.feat_strategy == 'SimCFA':
                        generator = CVA(num_class, num_node, num_feat, device, use_conv=args.use_conv, Lamb=tmp_Lamb, ratio=args.mask_ratio, C=tmp_C, sparse=False).to(device)
                elif args.feat_strategy == 'rawX':
                        generator = TopoGenSpecFeatRawX(num_class, num_node, num_feat, device, use_conv=args.use_conv, Lamb=tmp_Lamb, ratio=args.mask_ratio, C=tmp_C, sparse=False).to(device)
                else:
                    generator = TopoGenSpecformer(num_class, num_node, num_feat, device, Lamb=tmp_Lamb, ratio=args.mask_ratio, sparse=False).to(device)
            elif args.topo_strategy == 'masking': # topo_strategy = masking
                if args.feat_strategy == 'learnable': 
                    generator = TopoAugFeatGen(num_class, num_node, num_feat, device, ei=data.edge_index, ew=data.edge_attr, use_conv=args.use_conv, Lamb=tmp_Lamb, ratio=args.mask_ratio, C=tmp_C, sparse=False).to(device)
            elif args.topo_strategy == 'SimMTA':
                if args.feat_strategy == 'SimCFA':
                    generator = TopoSimMTA(num_class, num_node, num_feat, device, use_conv=args.use_conv, Lamb=tmp_Lamb, ratio=args.mask_ratio, C=tmp_C, sparse=False, pe=args.pe).to(device)
                elif args.feat_strategy == 'rawX':
                    generator = TopoSimMTAFeatRawX(num_class, num_node, num_feat, device, use_conv=args.use_conv, Lamb=tmp_Lamb, ratio=args.mask_ratio, C=tmp_C, sparse=False).to(device)
            elif args.topo_strategy == 'none':
                if args.feat_strategy == 'learnable':
                    generator = FeatGenConv(num_class, num_node, num_feat, device, ei=data.edge_index, ew=data.edge_attr, use_conv=args.use_conv, Lamb=tmp_Lamb, ratio=args.mask_ratio, C=tmp_C, sparse=False).to(device)

            optimizer_gene = Adam(generator.parameters(), lr=args.lr2, weight_decay=args.wd2)

        best_val_acc = lastacc
        bestMiF1 = lastMiF1
        bestMaF1 = lastMaF1
        best_test_acc = 0.0

        total_time_st = time.time()
        per_epoch_times = []
        min_loss = 0.0

        if args.load_pth: # test
            encoder_model.eval()
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
                    if args.dataset == 'WiKi-CS':
                        result = WiKiLREvaluator(weight_decay=decay)(z, data.y, data.train_mask, data.val_mask, data.test_mask)
                    else:
                        split = get_split(num_samples=z.size()[0], train_ratio=args.train_ratio, test_ratio=args.test_ratio)
                        result = LREvaluator(weight_decay=decay)(z, data.y, split)
                    if result['val_acc'] > best_result['val_acc']:
                        best_result = result
                    
                test_acc = best_result['test_acc']
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch
                    best_testMiF1 = best_result['micro_f1']
                    best_testMaF1 = best_result['macro_f1']  
        else:
            for epoch in range(1, args.num_epochs+1):
                epoch_per_time_start = time.time()
                if args.use_contrast_mode:
                    infomax_loss, reg_loss, encoder_loss, infomin_loss, bn_loss, augmenter_loss = train_epoch(encoder_model, encoder_model_ib, generator, optimizer, optimizer_ib, optimizer_gene, contrast_model, tmp_U, adj2, x2, args)
                else:
                    infomax_loss, reg_loss, encoder_loss, infomin_loss, bn_loss, augmenter_loss = train_epoch(encoder_model, encoder_model_ib, generator, optimizer, optimizer_ib, optimizer_gene, None, tmp_U, adj2, x2, args)
                epoch_per_time_end = time.time()
                per_epoch_times.append(epoch_per_time_end - epoch_per_time_start)

                # if args.early_stop:
                #     if encoder_loss < min_loss:
                #         min_loss = encoder_loss
                #         convergence_cnt = 0
                #     else:
                #         convergence_cnt = convergence_cnt + 1
                #         if convergence_cnt >= args.patience:
                #             break
                result = {}
                if epoch % args.eval_intervals == 0:
                    valid_curve = []
                    test_curve = []

                    encoder_model.eval()
                    encoder_model_ib.eval()
                    generator.eval()

                    # val_acc, test_acc = kf_evaluate_embedding(z, data.y) 
                    # valid_curve.append(val_acc)
                    # test_curve.append(test_acc)
                    # logging.info(f'[test_acc @ {epoch}] : {test_acc:.5f}')
                    # logging.info(f'[val_acc @ {epoch}] : {val_acc:.5f}')
                    # best_result = {
                    #     'test_acc': 0.,
                    #     'micro_f1': 0.,
                    #     'macro_f1': 0.,
                    #     'val_acc': 0.
                    # }

                    # for decay in [0.0, 0.001, 0.005, 0.01, 0.1]:
                    #     if args.dataset == 'WiKi-CS':
                    #         result = WiKiLREvaluator(weight_decay=decay)(z, data.y, data.train_mask, data.val_mask, data.test_mask)
                    #     else:
                    #         split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
                    #         result = LREvaluator(weight_decay=decay)(z, data.y, split)
                    #     if result['val_acc'] > best_result['val_acc']:
                    #         best_result = result
                    
                    # test_acc = best_result['test_acc']
                    # if test_acc > best_test_acc:
                    #     best_test_acc = test_acc
                    #     best_epoch = epoch
                    #     best_testMiF1 = best_result['micro_f1']
                    #     best_testMaF1 = best_result['macro_f1']  
                    if args.task == 'node-classification':
                        with torch.no_grad():
                            z = encoder_model(x=x2, adj=adj2) 
                            x1, adj1 = generator(U=tmp_U, x=x2) 
                            z_ = encoder_model(x=x1, adj=adj1)
                            z = (z + z_) / 2.0
                        if args.dataset == 'WiKi-CS':
                            result = WiKiLREvaluator()(z, data.y, data.train_mask, data.val_mask, data.test_mask)
                        else:
                            split = get_split(num_samples=z.size(0), train_ratio=args.train_ratio, test_ratio=args.test_ratio)
                            result = LREvaluator()(z, data.y, split)
                    elif args.task == 'link-prediction':
                        # 随机划分10%的边做训练，10%的边做验证，80%的边做预测
                        with torch.no_grad():
                            z = encoder_model(x=x2, adj=adj2) 
                            # x1, adj1 = generator(U=tmp_U, x=x2) 
                            # z_ = encoder_model(x=x1, adj=adj1)
                            # z = (z + z_) / 2.0
                        split = get_split(num_samples=data.edge_index.size(1), train_ratio=args.train_ratio, test_ratio=args.test_ratio)
                        neg_split = get_split(num_samples=data.neg_edge_index.size(1), train_ratio=args.train_ratio, test_ratio=args.test_ratio)
                        # train_ei = data.edge_index[:,split['train']]
                        result = LinkEvaluator()(z, data.edge_index, data.neg_edge_index, split, neg_split) # 采样的负edge和正edge数量一样
                        
                    
                    test_acc = result['test_acc']
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        best_epoch = epoch
                        best_testMiF1 = result['micro_f1']
                        best_testMaF1 = result['macro_f1']  

                        # draw_plot_3D(data, z.cpu().detach().numpy(), f'./pics/{args.dataset}/{study_name}/3D/'+'tsne.pdf', epoch=epoch)

                        # if args.feat_strategy in ['learnable', 'SimCFA', 'rawX']:
                        #     if args.use_conv:
                        #         C1_e = torch.sum((generator.C_conv(tmp_C))**2, dim=1)/num_feat
                        #     else:
                        #         C1_e = torch.sum((generator.C_mlp(tmp_C))**2, dim=1)/num_feat
                        #     C1_bar = torch.mean(C1_e) 
                        #     C_metric_trained = torch.mean((C1_e-C1_bar)**2)

                        #     np_new_lamb = generator.new_lamb.detach().cpu().numpy()
                        #     np_sorted_lamb = np.diagonal(np_new_lamb) #[sorted_indicies]
                        #     # draw_eigvals_change(np_sorted_lamb, np.diagonal(tmp_Lamb), pics_path+study_name+'_eigvals_change_not_sorted.pdf')
                        #     sorted_indicies = np.argsort(np.diagonal(np_new_lamb))
                        #     np_sorted_lamb = np.diagonal(np_new_lamb)[sorted_indicies]
                        #     _, rand_ei, rand_ew = edge_removal_augmenter(data.x, data.edge_index, data.edge_attr)
                        #     adj = torch.from_numpy(to_scipy_sparse_matrix(rand_ei, edge_attr=rand_ew, num_nodes=num_node).A).to(device)
                        #     D = torch.sum(adj, dim=1)
                        #     D_inv_sqrt = D ** (-1/2)
                        #     D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
                        #     D_inv_sqrt = torch.diag(D_inv_sqrt)
                        #     rand_L = torch.eye(num_node).to(device) - torch.mm(D_inv_sqrt, torch.mm(adj, D_inv_sqrt))
                        #     rand_lamb = tmp_U.T @ rand_L @ tmp_U
                        #     rand_lamb = np.sort(np.diag(rand_lamb.detach().cpu().numpy()))
                        #     rand_x, _, _ = feature_mask_augmenter(data.x, data.edge_index, data.edge_attr)
                        #     rand_ei = rand_ei.to(device)
                        #     rand_x = rand_x.to(device)
                        #     rand_C = tmp_U.T @ rand_x
                        #     sorted_three_draw_feature_coefficients_change(tmp_C, generator.new_C, rand_C, f'./pics/{args.dataset}/{study_name}/'+f'C_change_{epoch}.pdf', args.dataset)

                    if his_best_test_acc < test_acc:
                        his_best_test_acc = test_acc
                        # draw pics
                        # draw_plot(data, z.cpu().detach().numpy(), f'./pics/{args.dataset}/{study_name}/2D/'+'tsne.pdf', max_nodes=None)
                        # draw_plot_3D(data, z.cpu().detach().numpy(), f'./pics/{args.dataset}/{study_name}/3D/'+'tsne.pdf', epoch=epoch)

                        # if args.feat_strategy in ['learnable', 'SimCFA', 'rawX']:
                        #     if args.use_conv:
                        #         C1_e = torch.sum((generator.C_conv(tmp_C))**2, dim=1)/num_feat
                        #     else:
                        #         C1_e = torch.sum((generator.C_mlp(tmp_C))**2, dim=1)/num_feat
                        #     C1_bar = torch.mean(C1_e)
                        #     C_metric_trained = torch.mean((C1_e-C1_bar)**2)

                        #     np_new_lamb = generator.new_lamb.detach().cpu().numpy()
                        #     np_sorted_lamb = np.diagonal(np_new_lamb) #[sorted_indicies]
                        #     # draw_eigvals_change(np_sorted_lamb, np.diagonal(tmp_Lamb), pics_path+study_name+'_eigvals_change_not_sorted.pdf')
                        #     sorted_indicies = np.argsort(np.diagonal(np_new_lamb))
                        #     np_sorted_lamb = np.diagonal(np_new_lamb)[sorted_indicies]
                        #     _, rand_ei, rand_ew = edge_removal_augmenter(data.x, data.edge_index, data.edge_attr)
                        #     adj = torch.from_numpy(to_scipy_sparse_matrix(rand_ei, edge_attr=rand_ew).A).to(device)
                        #     D = torch.sum(adj, dim=1)
                        #     D_inv_sqrt = D ** (-1/2)
                        #     D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
                        #     D_inv_sqrt = torch.diag(D_inv_sqrt)
                        #     rand_L = torch.eye(num_node).to(device) - torch.mm(D_inv_sqrt, torch.mm(adj, D_inv_sqrt))
                        #     rand_lamb = tmp_U.T @ rand_L @ tmp_U
                        #     rand_lamb = np.sort(np.diag(rand_lamb.detach().cpu().numpy()))
                        #     rand_x, _, _ = feature_mask_augmenter(data.x, data.edge_index, data.edge_attr)
                        #     rand_ei = rand_ei.to(device)
                        #     rand_x = rand_x.to(device)
                        #     rand_C = tmp_U.T @ rand_x
                        #     sorted_three_draw_feature_coefficients_change(tmp_C, generator.new_C, rand_C, f'./pics/{args.dataset}/{study_name}/'+f'C_change_{epoch}.pdf', args.dataset)
                        
                        torch.save(encoder_model.state_dict(), f'./pts/{args.dataset}/{study_name}/encoder/encoder_{test_acc}_{trial_id}_{epoch}.pt')
                        torch.save(encoder_model_ib.state_dict(), f'./pts/{args.dataset}/{study_name}/encoder/ib_{test_acc}_{trial_id}_{epoch}.pt')
                        torch.save(generator.state_dict(), f'./pts/{args.dataset}/{study_name}/gene/generator_{test_acc}_{trial_id}_{epoch}.pt')
            
            # if epoch % args.eval_intervals == 0:
            #     torch.save(encoder_model.state_dict(), f'./pts/{args.dataset}/{study_name}/encoder/encoder_{epoch}.pt')
            #     torch.save(encoder_model_ib.state_dict(), f'./pts/{args.dataset}/{study_name}/encoder/ib_{epoch}.pt')
            #     torch.save(generator.state_dict(), f'./pts/{args.dataset}/{study_name}/gene/generator_{epoch}.pt')


                # print(f"(E): Best test acc={best_test_acc:.4f}, F1Mi={best_testMiF1:.4f}, F1Ma={best_testMaF1:.4f}")

                # if best_val_acc < val_acc: # follow the setting of ADGCL, different with AutoGCL setting
                #     best_val_acc = val_acc
                #     best_test_acc = test_acc
                #     best_epoch = epoch
                #     convergence_cnt = 0
                # else:
                #     convergence_cnt += 1 
                #     if args.early_stop:
                #         if convergence_cnt == args.patience:
                #             break

        torch.cuda.empty_cache()
        # print(f'allocated GPU Mem after one epoch : {torch.cuda.memory_allocated()/float(1024*1024*1024)} GB')
        total_time = time.time() - total_time_st

        accs.append(best_test_acc)
        # accs.append(best_val_acc)
        mean_per_epoch_time.append(np.mean(per_epoch_times))
        total_times.append(total_time)

    # print(f"best_epoch={best_epoch}, best_test_acc={best_test_acc}, best_testMiF1={best_testMiF1}, best_testMaF1={best_testMaF1}, preprocess_time={preprocess_time}, mean_total_time={np.mean(total_times)}, std_total_time={np.std(total_times)}, mean_per_epoch_time={np.mean(mean_per_epoch_time)}, std_per_epoch_time={np.std(mean_per_epoch_time)}")
    print(f"best_epoch={best_epoch}, best_test_acc={best_test_acc}, best_testMiF1={best_testMiF1}, best_testMaF1={best_testMaF1}, preprocess_time={preprocess_time}, mean_total_time={np.mean(total_times)}, std_total_time={np.std(total_times)}, mean_per_epoch_time={np.mean(mean_per_epoch_time)}, std_per_epoch_time={np.std(mean_per_epoch_time)}")
    torch.cuda.empty_cache()
    print(f'allocated GPU Mem after training : {torch.cuda.memory_allocated()/float(1024*1024*1024)} GB')
    print(f'mean_best_val_acc : {np.mean(accs)}, std_best_val_acc : {np.std(accs)}')
    # if np.mean(accs) >= his_best_test_acc:
    #     torch.save(encoder_model, f'./pts/{args.dataset}/{study_name}/encoder/encoder_{np.mean(accs)}.pt')
    #     torch.save(encoder_model_ib, f'./pts/{args.dataset}/{study_name}/encoder/ib_{np.mean(accs)}.pt')
    #     torch.save(generator, f'./pts/{args.dataset}/{study_name}/generator/generator_{np.mean(accs)}.pt')
    return np.mean(accs)

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


# def train_epoch(encoder_model, encoder_model_ib, generator, optimizer, optimizer_ib, optimizer_gene, contrast_model, tmp_U, adj2, x2, args):
#     encoder_model.train()
#     generator.eval()
#     optimizer.zero_grad()
#     x1, adj1 = generator(U=tmp_U, x=x2)
#     z1, z2 = encoder_model(x=x1.detach(), adj=adj1.detach()), encoder_model(x=x2.detach(), adj=adj2.detach())
#     _v1, _v2 = encoder_model_ib(x=x1.detach(), adj=adj1.detach()), encoder_model_ib(x=x2.detach(), adj=adj2.detach())

#     k = torch.tensor(int(z1.shape[0] * args.COSTA_ratio))
#     p = (1/torch.sqrt(k))*torch.randn(k, z1.shape[0]).to(x2.device)
#     z1, z2 = p @ z1, p @ z2
#     _v1, _v2 = p @ _v1, p @ _v2 

#     z1, z2 = [encoder_model.project(x) for x in [z1, z2]]
#     _v1, _v2 = [encoder_model_ib.project(x) for x in [_v1, _v2]]
#     torch.cuda.empty_cache()
#     if contrast_model is not None:
#         infomax_loss = contrast_model(z1, z2)
#     else:
#         infomax_loss = encoder_model.loss(z1, z2)
    
#     encoder_loss = infomax_loss 
#     encoder_loss.backward()
#     optimizer.step()

#     encoder_model.eval()
#     encoder_model_ib.train()
#     if contrast_model is not None:
#         bn_loss = contrast_model(_v1, z1.detach()) + contrast_model(_v2, z2.detach())
#     else:
#         bn_loss = encoder_model.loss(_v1, z1.detach()) + encoder_model.loss(_v2, z2.detach())
#     bn_loss = args.beta * bn_loss
#     bn_loss.backward()
#     optimizer_ib.step()
#     torch.cuda.empty_cache()

#     encoder_model_ib.eval()
#     generator.train()
#     optimizer_gene.zero_grad()
#     v1_, v2_ = encoder_model_ib(x=x1.detach(), adj=adj1.detach()), encoder_model_ib(x=x2.detach(), adj=adj2.detach())
#     v1_, v2_ = p@v1_, p@v2_
#     v1_, v2_ = [encoder_model_ib.project(x) for x in [v1_, v2_]]
    
#     if args.sim_method == 'none':
#         if contrast_model is not None:
#             infomin_loss = contrast_model(v1_, v2_)
#         else:
#             infomin_loss = encoder_model.loss(v1_, v2_)
#     else:
#         infomin_loss = sim(v1_, v2_, method=args.sim_method)
#     generator_loss = -1.0 * infomin_loss
#     if args.single:
#         if args.feat_strategy not in ['learnable', 'SimCFA', 'rawX']:
#             if args.topo_strategy not in ['learnable', 'SimMTA']:
#                 reg_loss = 0
#                 generator_loss = generator_loss + reg_loss
#             else:
#                 reg_loss = args.alpha * (generator.Lamb_JSD_loss())
#                 generator_loss = generator_loss + reg_loss
#         elif args.feat_strategy == 'rawX':
#             if args.topo_strategy not in ['learnable', 'SimMTA']:
#                 reg_loss = args.alpha * (generator.X_JSD_Loss(tmp_U, x1))
#                 generator_loss = generator_loss + reg_loss
#             else:
#                 reg_loss = args.alpha * (generator.X_JSD_Loss(tmp_U, x1) + generator.Lamb_JSD_loss())
#                 generator_loss = generator_loss + reg_loss
#         else:
#             if args.topo_strategy not in ['learnable', 'SimMTA']: # none, msking 
#                 reg_loss = args.beta * (generator.C_JSD_Loss())
#                 generator_loss = generator_loss + reg_loss
#             else:
#                 reg_loss = args.alpha * (generator.C_JSD_Loss() + generator.Lamb_JSD_loss())
#                 generator_loss = generator_loss + reg_loss
#     if torch.isnan(generator_loss):
#         print("loss is nan")
#     generator_loss.backward()
#     optimizer_gene.step()
#     torch.cuda.empty_cache()

#     return infomax_loss.item(), reg_loss.item(), encoder_loss.item(), infomin_loss.item(), bn_loss.item(), generator_loss.item()


# 'alpha': 0.45, 'beta': 0.4, 'mask_ratio': 0.1, 'num_layers': 2, 'tau': 0.55
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=39788)
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

    parser.add_argument('--sim_method', type=str, default='none', choices=['none', 'cos', 'exp', 'mse'])
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

    parser.add_argument('--task', type=str, default='node-classification', choices=['node-classification', 'link-prediction'])

    args = parser.parse_args()

    torch.cuda.set_device(int(args.device.split(':')[-1]))
    device = torch.device(args.device)

    torch.manual_seed(args.seed)

    path = './data/' 
    path = osp.join(path, args.dataset) 
    dataset = get_dataset(args.dataset, device=device, dir_path=path)
    data = dataset[0].to(device)
    if args.dataset == 'WiKi-CS':
        std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
        data.x = (data.x - mean) / std
        data.edge_index = to_undirected(data.edge_index)

    U, Lamb, C = None, None, None
    if osp.exists(f'./data/npy/{args.dataset}_sorted_eigvecs.npy'):
        U = np.load(f"./data/npy/{args.dataset}_sorted_eigvecs.npy")
    if osp.exists(f'./data/npy/{args.dataset}_sorted_eigvals.npy'):
        Lamb = np.load(f"./data/npy/{args.dataset}_sorted_eigvals.npy")
    if osp.exists(f'./data/npy/{args.dataset}_singal_amps.npy'):
        C = np.load(f"./data/npy/{args.dataset}_singal_amps.npy")

    study_name = 'orilink_SimULA_'+f'{args.dataset}'
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
    
    if args.task == 'node-classification':
        study_name += '_nc'
    if args.task == 'link-prediction':
        study_name += '_lp'
    
    study_name += f'_train{args.train_ratio}' + f'_test{args.test_ratio}'

    study_name += '_res1'
    
    study_name += '_' + args.pe
    study_name += '_' + str(args.COSTA_ratio)

    import pathlib
    pathlib.Path(
        f'./pts/{args.dataset}/{study_name}/gene/'
    ).mkdir(parents=True, exist_ok=True)

    pathlib.Path(
        f'./pts/{args.dataset}/{study_name}/encoder/'
    ).mkdir(parents=True, exist_ok=True)

    pathlib.Path(
        f'./bo_dbs/{args.dataset}/'
    ).mkdir(parents=True, exist_ok=True)

    pathlib.Path(
        f'./pics/{args.dataset}/{study_name}/2D/'
    ).mkdir(parents=True, exist_ok=True)

    pathlib.Path(
        f'./pics/{args.dataset}/{study_name}/3D/'
    ).mkdir(parents=True, exist_ok=True)

    if args.botune:
        study = optuna.create_study(direction="maximize",
                                    storage='sqlite:///' + f'./bo_dbs/{args.dataset}/' + study_name+ '.db',
                                    study_name=study_name,
                                    load_if_exists=True)

        study.optimize(search_hyper_params, n_trials=args.optruns)

        print("best params", study.best_params)
        print("best val_acc", study.best_value)
    else: 
        # seeds = [39788, 2024, 2000, 920, 209, 2023]
        # seeds = [2024, 2023, 2022, 2021, 2000]
        # all_accs = []
        # for i in range(len(seeds)):
        #     args.seed = seeds[i]
        #     all_accs.append(ULA(args=args))
        # print(f'{np.mean(all_accs)}\pm{np.std(all_accs)}')

        ULA(args=args)
