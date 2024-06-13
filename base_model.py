import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class SineEncoding(nn.Module):
    def __init__(self, hidden_him=128):
        super(SineEncoding, self).__init__()
        self.hidden_him = hidden_him
        self.constant = 100
                                                                            
        self.eig_w = nn.Linear(hidden_him+1, hidden_him, bias=False)         

    def forward(self, eigvals):
        '''
            input: [N]
            output: [N, d]
        '''

        eigvals_emb = eigvals * self.constant
        div = torch.exp(torch.arange(0, self.hidden_him, 2) * (-math.log(1000)/self.hidden_him)).to(eigvals.device)
        pos_emb = eigvals_emb.unsqueeze(1) * div 
        eeig = torch.cat((eigvals.unsqueeze(1), torch.sin(pos_emb), torch.cos(pos_emb)), dim=1) 

        return self.eig_w(eeig)

class NoEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(NoEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.constant = 100
        self.eig_w = nn.Linear(1, hidden_dim, bias=False)

    def forward(self, eigvals):
        eeig = eigvals * self.constant
        eeig = eeig.unsqueeze(1)
        return self.eig_w(eeig)

class FFN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim) -> None:
        super(FFN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hid_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hid_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
    
class SpecLayer(nn.Module):
    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none') -> None:
        super(SpecLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)

        if norm == 'none':
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == 'LN':
            self.norm = nn.LayerNorm(ncombines)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(ncombines)
        else:
            self.norm = None

    def forward(self, x):

        x = self.prop_dropout(x) * self.weight # [N, m, d] * [1, m, d]
        x = torch.sum(x, dim=1)

        if self.norm is not None:
            x = self.norm(x)
            x = F.relu(x)

        return x

class SimplifySpecformer(nn.Module):
    # remove residual block
    def __init__(self, num_node, nclass, nfeat, num_layer=2, hid_dim=128, nheads=1, tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none', add_noise=False, resblk1=True, resblk2=True, device=torch.device('cpu'), pe='Sine') -> None:
        super(SimplifySpecformer, self).__init__()
        self.num_node = num_node
        self.nclass = nclass
        self.nfeat = nfeat
        self.nlayer = num_layer
        self.hid_dim = hid_dim
        self.nheads = nheads
        self.resblk1 = resblk1
        self.resblk2 = resblk2
        self.add_noise = add_noise
        
        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, nclass),
        )
        self.pe = pe 
        # for arxiv & penn dataset
        # self.linear_encoder = nn.Linear(nfeat, hid_dim)
        # self.classify = nn.Linear(hid_dim, nclass)

        if pe == 'Sine':
            self.eig_encoder = SineEncoding(hid_dim) # Eigen_Embedder
        else:
            self.eig_encoder = NoEncoding(hid_dim)
        
        self.decoder = nn.Linear(hid_dim, nheads)

        self.mha_norm = nn.LayerNorm(hid_dim)
        self.ffn_norm = nn.LayerNorm(hid_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hid_dim, nheads, tran_dropout)
        self.ffn = FFN(hid_dim, hid_dim, hid_dim)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        self.prop_dropout = nn.Dropout(prop_dropout)

        # self.bias = nn.Parameter(torch.randn((num_node, hid_dim)), requires_grad=True)

        self.norm = norm
        if norm == 'none':
            self.layers = nn.ModuleList([SpecLayer(nheads+1, nclass, prop_dropout, norm=norm) for i in range(self.nlayer)])
        
        self.I = torch.ones(self.num_node).to(device)

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, FFN):
                m.reset_parameters()
        

    def forward(self, eigvals, eigvecs, x):
        N = eigvals.size(0)
        U_T = eigvecs.permute(1, 0)     # 

        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        # else:
            # h = self.feat_dp1(x)
            # h = self.linear_encoder(h)

        eig_emb = self.eig_encoder(eigvals)
        # print(f"eig_emb.size : {eig_emb.size}")

        mha_eig_emb = self.mha_norm(eig_emb)
        mha_eig_emb, attn = self.mha(mha_eig_emb, mha_eig_emb, mha_eig_emb)
        # residual block 1 
        if self.resblk1:
            eig_emb = eig_emb + self.mha_dropout(mha_eig_emb)

        ffn_eig_emb = self.ffn_norm(eig_emb)
        ffn_eig_emb = self.ffn(ffn_eig_emb)
        # residual block 2
        if self.resblk2:
            eig_emb = eig_emb + self.ffn_dropout(ffn_eig_emb)

        assert eig_emb.size(0) == self.num_node
        assert eig_emb.size(1) == self.hid_dim

        # if self.add_noise:
            # eig_emb = eig_emb + self.bias 
        new_eigvals = self.decoder(eig_emb)
        # print("new_eigvals.shape", new_eigvals.size())
        # for conv in self.layers:
        #     basic_feats = [h]
        #     U_T_X = U_T @ h
        #     for i in range(self.nheads):
        #         basic_feats.append(eigvecs @ (new_eigvals[:, i].unsqueeze(1) * U_T_X))  # [N, d]  # basidas
        #     basic_feats = torch.stack(basic_feats, axis=1)                # [N, m, d]
        #     h = conv(basic_feats)

        new_eigvals_diags = torch.mean(new_eigvals, dim=-1)
        # new_eigvals_diags = torch.diag(torch.mean(new_eigvals, dim=-1)) 
        # new_eigvals_diags = F.relu(new_eigvals_diags) 
        new_eigvals_diags = F.relu(torch.tanh(new_eigvals_diags) + self.I)  
        return new_eigvals_diags

class AdaptiveConvs(nn.Module):
    def __init__(self, d):
        super(AdaptiveConvs, self).__init__()
        self.d = d
        self.convs = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if self.d == x.size(1):
            x = x.unsqueeze(0) # (N, C) => (B, N, C)
            x = x.transpose(1,2).unsqueeze(3) # (B, N, C) => (B, C, N, 1) 
            x = self.convs(x)
            # x = self.pool(x)
            x = x.squeeze(3).transpose(1,2).squeeze(0) # (B, C, N, 1) => (B, N, C) => (N, C)
        elif self.d == x.size(0):
            x = x.transpose(0,1).unsqueeze(0) # (N, C) => (C, N) => (B, N, C)
            x = x.transpose(1,2).unsqueeze(3) # (B, C, N) => (B, N, C, 1)
            x = self.convs(x)
            # x = self.pool(x)
            x = x.squeeze(3).transpose(1,2).squeeze(0) # (B, N, C, 1) => (B, C, N) => (C, N)
            x = x.transpose(0,1) # (C, N) => (N, C)

        return x


class SimplifyConvs(nn.Module):
    def __init__(self, d) -> None:
        super(SimplifyConvs, self).__init__()
        self.d = d 
        self.weights = nn.Parameter(torch.randn(d,d), requires_grad=True)

    def forward(self, x):
        if self.d == x.size(1):
            num_samples = x.size(0)
            # accelerate the computation
            x = torch.matmul(x, self.weights) # d x N => N x d
                # x = torch.mul(x, self.weights.unsqueeze(0)).transpose(1,2).sum(-1) # .sum(dim=-1)
            # else:
            #     x = x.unsqueeze(-1) # (N, C) => (N, C, 1)
            #     x_s = []
            #     for i in range(num_samples):
            #         x_s.append(torch.sum(torch.mul(x[i], self.weights),dim=-2))
            #     x = torch.stack(x_s, dim=0)
            # print(f"when apply SimplifyConvs, new_x.size() = {x.size()}")
            # x = torch.mul(x, self.weights)
            # x = torch.sum(x, dim=-2)
            # x = x.squeeze(1)
        return x

class CVA(nn.Module):
    def __init__(self, num_class, num_node, num_feat, device, use_conv=True, Lamb=None, ratio=0.0, C=None, sparse=False, num_layer=1, hidden_dim=128, nheads=1, tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none', add_noise=False, pe='Sine') -> None:
        super(CVA, self).__init__()
        self.specformer = SimplifySpecformer(num_node, num_class, num_feat, num_layer=num_layer, hid_dim=hidden_dim, nheads=nheads,
                tran_dropout=tran_dropout, feat_dropout=feat_dropout, prop_dropout=prop_dropout, norm=norm, add_noise=add_noise, device=device, pe=pe)
        self.num_class = num_class
        self.num_node = num_node
        self.num_feat = num_feat
        self.device = device
        self.Lamb = torch.diagonal(torch.from_numpy(Lamb).to(torch.float32)).to(device)
        self.ratio = ratio
        self.C = C.to(device)
        self.use_conv = use_conv
        if use_conv:
            self.C_conv = SimplifyConvs(num_feat) # 
            # self.C_conv=AdaptiveConvs(num_node)
        else:
            self.C_mlp = nn.Sequential(nn.Linear(num_feat, num_feat), nn.Tanh(), nn.Linear(num_feat, num_feat)) 
        self.I = torch.ones(self.num_node).to(device)
        if self.ratio != 0.0:
            mask_list = [0.0]*num_node
            if ratio != 0.0:
                fixed = int(num_node * ratio)
                for i in range(fixed):
                    mask_list[i] = 1.0
            self.fix_pos = torch.tensor(mask_list).to(device)
        
        self.reset_parameters()

    def reset_parameters(self):
        if not self.use_conv:
            for m in self.C_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def get_diag_mat(self, U):
        adj = torch.diag(self.I) - torch.mm(U, torch.mm(self.new_lamb, U.T))
        return adj 

    def forward(self, U, x=None):
        x = torch.mm(U, self.C)
        self.new_lamb = self.specformer(self.Lamb, U, x)
        if self.ratio:
            self.new_lamb = torch.mul((self.I - self.fix_pos), self.new_lamb) + torch.mul(self.fix_pos, self.Lamb)
            # normalized_new_lamb =  torch.diag(F.softmax(torch.sum(self.new_lamb, dim=1, keepdim=True), dim=0).view(-1))
            # normalized_lamb = torch.diag(F.softmax(torch.sum(self.Lamb, dim=1, keepdim=True), dim=0).view(-1))
            # diag_mat = torch.mm((self.I - self.fix_pos), torch.clamp(self.lamb_mlp(self.Lamb), 0, 2)) + torch.mm(self.fix_pos,self.Lamb)
        self.new_lamb = torch.diag(self.new_lamb)
        new_adj = self.get_diag_mat(U)
        if self.use_conv:
            self.new_C = self.C_conv(self.C)
        else:
            self.new_C = self.C_mlp(self.C)
        new_x = torch.mm(U, self.new_C)
        return new_x, new_adj

    def wasserstein_loss(self):
        wasserstein_loss = torch.norm(self.new_lamb - self.Lamb, p=1) / self.num_node # 
        return wasserstein_loss
    
    def Lamb_KL_loss(self): 
        # normalized_new_lamb =  F.softmax(torch.sum(self.new_lamb, dim=1, keepdim=True), dim=0).view(-1)
        normalized_new_lamb =  F.softmax(torch.sum(self.new_lamb, dim=1, keepdim=True), dim=0).view(-1)
        normalized_lamb = F.softmax(self.Lamb).view(-1)
        # normalized_lamb = F.softmax(torch.sum(self.Lamb, dim=1, keepdim=True), dim=0).view(-1)
        KL_loss_1 = F.kl_div(normalized_lamb.log(), normalized_new_lamb, reduction='batchmean')
        KL_loss_2 = F.kl_div(normalized_new_lamb.log(), normalized_lamb, reduction='batchmean')
        KL_loss = (KL_loss_1 + KL_loss_2) / 2.0
        return KL_loss # 
    
    def Lamb_JSD_loss(self):
        normalized_new_lamb =  F.softmax(torch.sum(self.new_lamb, dim=1, keepdim=True), dim=0).view(-1)
        normalized_lamb = F.softmax(self.Lamb).view(-1)
        JSD_loss_1 = 1e5 * F.kl_div(((normalized_new_lamb+normalized_lamb)/2.0).log(), normalized_lamb, reduction='batchmean')
        JSD_loss_2 = 1e5 * F.kl_div(((normalized_lamb+normalized_new_lamb)/2.0).log(), normalized_new_lamb, reduction='batchmean')
        JSD_loss = (JSD_loss_1 + JSD_loss_2) / 2.0  
        return JSD_loss
    
    def C_KL_Loss(self):
        normalized_new_C = F.log_softmax(self.new_C, dim=1)
        normlized_C = F.log_softmax(self.C, dim=1)
        KL_loss_1 = F.kl_div(normalized_new_C.log(), normlized_C, reduction='batchmean')
        KL_loss_2 = F.kl_div(normlized_C.log(), normalized_new_C, reduction='batchmean')
        KL_loss = (KL_loss_1 + KL_loss_2) / 2.0
        return KL_loss
    
    def C_JSD_Loss(self):
        normalized_new_C = F.softmax(self.new_C, dim=1)
        normalized_C = F.softmax(self.C, dim=1)
        JSD_loss_1 = 1e5 * F.kl_div(((normalized_new_C+normalized_C)/2.0).log(), normalized_new_C, reduction='batchmean')
        JSD_loss_2 = 1e5 * F.kl_div(((normalized_new_C+normalized_C)/2.0).log(), normalized_C, reduction='batchmean')
        JSD_loss = (JSD_loss_1 + JSD_loss_2) / 2.0
        return JSD_loss

