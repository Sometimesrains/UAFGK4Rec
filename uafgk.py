import pandas as pd
import numpy as np
import scipy.sparse as sp
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch_geometric.nn import HypergraphConv
from torch_geometric.data import Data

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class UAFGK(GeneralRecommender):
    def __init__(self, config, dataset):
        super(UAFGK, self).__init__(config, dataset)
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        df_uv = pd.read_csv(dataset_path+'/u_v_sim_graph.csv')
        df_ut = pd.read_csv(dataset_path+'/u_t_sim_graph.csv') 
        uv_matrix = self._coo_matrix(df_uv)
        ut_matrix = self._coo_matrix(df_ut) 
        
        # load parameters info
        self.latent_dim = config['embedding_size'] 
        self.n_layers = config['n_layers']  
        self.reg_weight = config['reg_weight']  
        self.dropout = 1-config['dropout']
        self.mask = config['mask']
        self.bpr_weight = config['bpr_weight']
        self.tc_weight = config['tc_weight'] 
        self.lc_weight = config['lc_weight'] 
        self.cvbpr_weight = config['cvbpr_weight'] 
        self.tight_tem= config['tight_tem'] 
        self.loose_tem= config['loose_tem'] 
        self.n_nodes = self.n_users + self.n_items

        # define layers and loss
        self.user_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.latent_dim)))
        self.item_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.latent_dim)))

        self.user_visual_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.latent_dim)))
        self.user_textual_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.latent_dim)))
        
        self.df_k = pd.read_csv(dataset_path+'/hypergraph.csv')
        hyper_edges = self.df_k[['row', 'col']].values.T
        n_k_nodes = self.df_k['col'].max() + 1
        k_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(int(n_k_nodes), self.latent_dim)))
        edge_index = torch.tensor(hyper_edges, dtype=torch.long)
        self.hypergraph_knowledge = Data(x=k_embeddings, edge_index=edge_index).to(self.device)

        self.user_visual_norm_adj = self.get_norm_adj_mat_uu(uv_matrix).to(self.device)
        self.user_textual_norm_adj = self.get_norm_adj_mat_uu(ut_matrix).to(self.device)
        
        # normalized adj matrix
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.masked_adj = None
        self.forward_adj = None
        self.pruning_random = False

        # edge prune
        self.edge_indices, self.edge_values = self.get_edge_info()

        self.mf_loss = BPRLoss()
        self.reg_loss = L2Loss()
        
        self.hyper_conv1 = HypergraphConv(in_channels=64, out_channels=64)
        self.hyper_conv2 = HypergraphConv(in_channels=64, out_channels=64)
        # self.hyper_conv3 = HypergraphConv(in_channels=64, out_channels=64)
        # self.hyper_conv4 = HypergraphConv(in_channels=64, out_channels=64)
        
        self.predictor = nn.Linear(self.latent_dim, self.latent_dim)
        self.predictor2 = nn.Linear(self.latent_dim, self.latent_dim)
        # nn.init.xavier_normal_(self.predictor.weight)
        nn.init.xavier_normal_(self.predictor2.weight)  # i_a and i_b use xavier, u_v and u_t use normal
    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj_matrix
            return

        keep_len = int(self.edge_values.size(0) * (self.dropout))

        def prune_edges(edge_indices, edge_values, keep_len, pruning_random):
            if pruning_random: 
                keep_idx = torch.tensor(random.sample(range(edge_values.size(0)), keep_len))
            else:
                keep_idx = torch.multinomial(edge_values, keep_len)  
            pruning_random = True ^ pruning_random
            keep_indices = edge_indices[:, keep_idx]
            return keep_indices, pruning_random

        keep_indices, self.pruning_random = prune_edges(self.edge_indices, self.edge_values, keep_len, self.pruning_random)
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj_matrix.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        ego_embeddings = torch.cat([self.user_embeddings, self.item_embeddings], 0)
        return ego_embeddings

    def forward(self):
        x, edge_index = self.hypergraph_knowledge.x, self.hypergraph_knowledge.edge_index
        x = self.hyper_conv1(x, edge_index)
        x = F.relu(x)
        x = self.hyper_conv2(x, edge_index)
        # x = F.relu(x)
        # x = self.hyper_conv3(x, edge_index)
        # x = F.relu(x)
        # x = self.hyper_conv4(x, edge_index)                
        hyperedge_embeddings = []
        for hyperedge in self.df_k['row'].unique():
            nodes = self.df_k[self.df_k['row'] == hyperedge]['col'].values
            hyperedge_embedding = x[nodes].mean(dim=0)
            hyperedge_embeddings.append(hyperedge_embedding)
        
        hyperedge_embeddings = torch.stack(hyperedge_embeddings)
        split_idx = hyperedge_embeddings.size(0) // 2
        ib_all_embeddings, ia_all_embeddings = hyperedge_embeddings[:split_idx], hyperedge_embeddings[split_idx:]
        
        
        
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = ego_embeddings

        ui_embeddings_layers = []
        uv_embeddings_layers = []
        ut_embeddings_layers = []

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.forward_adj, all_embeddings)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings)
            ui_embeddings_layers.append(all_embeddings)
            
            # u-u visual similarity graph GCN

            uv_embeddings = torch.sparse.mm(self.user_visual_norm_adj, self.user_visual_embeddings).to(self.device)
            uv_embeddings_layers.append(uv_embeddings)
            
            # u-u textual similarity graph GCN
            ut_embeddings = torch.sparse.mm(self.user_textual_norm_adj, self.user_textual_embeddings).to(self.device)
            ut_embeddings_layers.append(ut_embeddings)
            

        ui_all_embeddings = torch.sum(torch.stack(ui_embeddings_layers, dim=0), dim=0)
        user_all_embeddings, item_all_embeddings = torch.split(ui_all_embeddings, [self.n_users, self.n_items])
        uv_all_embeddings = torch.sum(torch.stack(uv_embeddings_layers, dim=0), dim=0)
        ut_all_embeddings = torch.sum(torch.stack(ut_embeddings_layers, dim=0), dim=0)
        
        return user_all_embeddings, item_all_embeddings,uv_all_embeddings,ut_all_embeddings,ib_all_embeddings, ia_all_embeddings

    def bpr_loss(self, u_embeddings, i_embeddings, user, pos_item, neg_item):
        u_embeddings = u_embeddings[user]
        posi_embeddings = i_embeddings[pos_item]
        negi_embeddings = i_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        m = torch.nn.LogSigmoid()
        bpr_loss = torch.sum(-m(pos_scores - neg_scores))
        #mf_loss = self.mf_loss(pos_scores, neg_scores)
        return bpr_loss

    def emb_loss(self, user, pos_item, neg_item):
        # calculate BPR Loss
        u_ego_embeddings = self.user_embeddings[user]
        posi_ego_embeddings = self.item_embeddings[pos_item]
        negi_ego_embeddings = self.item_embeddings[neg_item]

        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        return reg_loss

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        self.forward_adj = self.masked_adj
        e_u,e_i,e_u_v,e_u_t,e_i_b,e_i_a = self.forward()
        
        if self.mask >0:
            e_u = F.dropout(e_u, self.mask)
            e_i = F.dropout(e_i, self.mask)
        else:
            pass
        e_u_v, e_u_t = self.predictor(e_u_v), self.predictor(e_u_t)
        e_i_b, e_i_a = self.predictor2(e_i_b), self.predictor2(e_i_a)
        
        mf_loss = self.bpr_loss(e_u, e_i, user, pos_item, neg_item)
        
        L_bpr_vb = self.bpr_loss(e_u_v, e_i_b, user, pos_item, neg_item)
        L_bpr_va = self.bpr_loss(e_u_v, e_i_a, user, pos_item, neg_item)
        L_bpr_tb = self.bpr_loss(e_u_t, e_i_b, user, pos_item, neg_item)
        L_bpr_ta = self.bpr_loss(e_u_t, e_i_a, user, pos_item, neg_item)
        cv_loss = L_bpr_vb+L_bpr_va+L_bpr_tb+L_bpr_ta
        
        reg_loss = self.reg_loss(e_u[user], e_u_v[user],e_u_t[user],
                                 e_i[pos_item],e_i[neg_item],e_i_b[pos_item],e_i_a[pos_item],e_i_b[neg_item],e_i_a[neg_item])
        L_tc = self.t_contrastive_loss(e_u_v,e_u_t,user)
        L_lc = self.l_contrastive_loss(e_i_b,e_i_a,pos_item) 
        loss = self.bpr_weight*mf_loss+self.cvbpr_weight*cv_loss + self.reg_weight * reg_loss +self.tc_weight*L_tc+self.lc_weight*L_lc
        return loss, mf_loss,cv_loss,L_tc,L_lc
    
    def full_sort_predict(self, interaction):
        user = interaction[0]

        self.forward_adj = self.norm_adj_matrix
        e_u,e_i,e_u_v,e_u_t,e_i_b,e_i_a = self.forward()
        u_embeddings = e_u[user]
        u_embeddings_v =  e_u_v[user]
        u_embeddings_t =  e_u_t[user]
        
        scores1 = torch.matmul(u_embeddings, e_i.transpose(0, 1))
        scores2 = torch.matmul(u_embeddings_v, e_i_b.transpose(0, 1))
        scores3 = torch.matmul(u_embeddings_v, e_i_a.transpose(0, 1))
        scores4 = torch.matmul(u_embeddings_t, e_i_b.transpose(0, 1))
        scores5 = torch.matmul(u_embeddings_t, e_i_a.transpose(0, 1))   
        scores = scores1+scores2+scores3+scores4+scores5  
        return scores

    def _coo_matrix(self, dataframe):
        rows = dataframe['u1'].values
        cols = dataframe['u2'].values
        values = dataframe['label'].values
        coo_matrix = sp.coo_matrix((values, (rows, cols)))
        return coo_matrix
    
    def get_norm_adj_mat_uu(self,co_matrix):
        sim_M = co_matrix
        sim_M_t =  co_matrix.transpose()
        A = sim_M.dot(sim_M_t)
        A.setdiag(0)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)  
        L = D* A * D   
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_users, self.n_users)))
    
    def t_contrastive_loss(self, user_v, user_t, user):
        user_v = user_v[user]
        user_t = user_t[user]

        sim_pos = torch.sum(user_v * user_t, dim=1) / self.tight_tem
        
        sim_matrix = torch.matmul(user_v, user_t.T) / self.tight_tem
        
        exp_sim_matrix = torch.exp(sim_matrix)
        exp_sim_pos = torch.exp(sim_pos)
        
        mask = torch.eye(exp_sim_matrix.size(0), device=exp_sim_matrix.device).bool()
        
        exp_sim_matrix = exp_sim_matrix.masked_fill(mask, 0)
    
        softmax_denominator = exp_sim_matrix.sum(dim=1)
        softmax_numerator = exp_sim_pos
    
        contrastive_loss = -torch.log(softmax_numerator / softmax_denominator).sum()

        return contrastive_loss
    
    def l_contrastive_loss(self, item_b, item_a, pos_item):
        item_b = item_b[pos_item]
        item_a = item_a[pos_item]

        sim_pos = torch.sum(item_b * item_a, dim=1) / self.loose_tem
        
        sim_matrix = torch.matmul(item_b, item_a.T) / self.loose_tem
        
        exp_sim_matrix = torch.exp(sim_matrix)
        exp_sim_pos = torch.exp(sim_pos)
        
        mask = torch.eye(exp_sim_matrix.size(0), device=exp_sim_matrix.device).bool()
        
        exp_sim_matrix = exp_sim_matrix.masked_fill(mask, 0)
        
        softmax_denominator = exp_sim_matrix.sum(dim=1)
        softmax_numerator = exp_sim_pos
        
        contrastive_loss = -torch.log(softmax_numerator / softmax_denominator).sum()

        return contrastive_loss