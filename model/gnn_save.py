import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from .gnn_conv import GCNConv
from torch.nn import Dropout, Linear, BatchNorm1d
# from torch_geometric.nn import GCNConv as GCN


class GcnMlp(nn.Module):
    def __init__(self, in_dim, mid_dim, las_dim, dropout):
        super(GcnMlp, self).__init__()
        self.fc1 = Linear(in_dim, mid_dim)
        self.fc2 = Linear(mid_dim, las_dim)
        self.Act1 = nn.ReLU()
        self.Act2 = nn.ReLU()
        self.reset_parameters()
        self.dropout = dropout
        self.BNorm0 = BatchNorm1d(in_dim, eps=1e-5)
        self.BNorm1 = BatchNorm1d(mid_dim, eps=1e-5)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-5)
        nn.init.normal_(self.fc2.bias, std=1e-5)

    def forward(self, x):
        x = self.Act1(self.fc1(self.BNorm0(x)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.Act2(self.fc2(self.BNorm1(x)))
        return x


class MyGCN(torch.nn.Module):
    def __init__(self, infeat, outfeat, nclass, nROI, upper_feat_dim=1000, nhid=32, dropout=0.2,
                 weight_decay=5e-4,
                 with_relu=True,
                 device=None):
        super(MyGCN, self).__init__()
        self.device = device
        self.nclass = nclass
        self.dropout = dropout
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        # MLP layer
        self.mid = 256
        self.last = 32
        # Graph Convolutational layer
        self.infeat = infeat
        self.nhid = 64
        self.outfeat = 16  # outfeat
        self.fir_upperfea = upper_feat_dim
        self.nROI = nROI

        self.upper_mlp = MLP(self.fir_upperfea, hidden_size=512, hidden_size2=128, num_classes=nclass)
        self.conv1 = GCNConv(self.infeat, self.nhid, bias=True)
        self.conv2 = GCNConv(self.nhid, self.outfeat, bias=True)
        self.mlp = GcnMlp(self.nhid * 2 + self.outfeat * 2 + 128 + 64, self.mid, self.last, 0.8)  # 全脑+半脑+MLP
        # self.mlp = GcnMlp(self.nhid * 2 + self.outfeat * 2 + 128, self.mid, self.last, dropout)  # 全脑+MLP
        # self.mlp = GcnMlp(128 + 64, self.mid, self.last, dropout)  # 半脑+MLP
        # self.mlp = GcnMlp(128, self.mid, self.last, dropout)  # 去除全脑和半脑，只留上三角MLP
        # self.mlp = GcnMlp(self.nhid * 2 + self.outfeat * 2, self.mid, self.last, dropout)  # 只有全脑
        # self.mlp = GcnMlp(64, self.mid, self.last, dropout)  # 只有半脑
        # self.mlp = GcnMlp(self.nhid * 2 + self.outfeat * 2 + 64, self.mid, self.last, dropout)  # 全脑+半脑
        self.classifier = Linear(self.last, nclass)

        self.harf_bipartite_network = HarfBipartiteNetwork(self.infeat, self.nROI, dropout=self.dropout)

    def forward(self, x, edge_index, batch, edge_attr, upper_feat, data):
        harf_bipartite_features, bipartite_x_readout1, bipartite_x_readout2,\
            left_consistency1, right_consistency1, left_consistency2, right_consistency2 = self.harf_bipartite_network(batch, data)

        Upper_feat = upper_feat.to('cuda')
        _, outfeat_upper = self.upper_mlp(Upper_feat)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x_readout1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x_readout2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x_feat = torch.cat([x_readout1, x_readout2, outfeat_upper, harf_bipartite_features], dim=1)  # 全脑+半脑+MLP
        # x_feat = torch.cat([x_readout1, x_readout2, outfeat_upper], dim=1)  # 全脑+MLP
        # x_feat = torch.cat([outfeat_upper, harf_bipartite_features], dim=1)  # 半脑+MLP
        # x_feat = outfeat_upper  # 去除全脑和半脑，只留上三角MLP
        # x_feat = torch.cat([x_readout1, x_readout2], dim=1)  # 只有全脑
        # x_feat = harf_bipartite_features  # 只有半脑
        # x_feat = torch.cat([x_readout1, x_readout2, harf_bipartite_features], dim=1)  # 全脑+半脑
        x_feat = self.mlp(x_feat)
        logits = self.classifier(x_feat)
        classfication = F.log_softmax(logits, dim=1)

        distances1 = F.pairwise_distance(bipartite_x_readout1, x_readout1, p=2)
        distances2 = F.pairwise_distance(bipartite_x_readout2, bipartite_x_readout2, p=2)
        consistency_loss = (torch.mean(distances1) + torch.mean(distances2))/2  # 一致性约束损失
        # consistency_loss = 0
        harf_consistency_loss = (torch.mean(left_consistency1) + torch.mean(right_consistency1) +
                                 torch.mean(left_consistency2) + torch.mean(right_consistency2))/4

        return classfication, consistency_loss, harf_consistency_loss, x_feat, logits


class HarfBipartiteNetwork(torch.nn.Module):
    def __init__(self, infeat, nROI, dropout=0.8):
        super(HarfBipartiteNetwork, self).__init__()
        self.dropout = dropout
        # MLP layer
        self.mid = 256
        self.last = 32
        # Graph Convolutational layer
        self.infeat = infeat
        self.nhid = 64
        self.nhid2 = 32
        self.outfeat = 16  # outfeat
        self.nROI = nROI

        self.conv_left1 = GCNConv(self.infeat, self.nhid, bias=True)
        self.conv_left2 = GCNConv(self.nhid * 2, self.nhid2, bias=True)
        self.conv_left3 = GCNConv(self.nhid2, self.outfeat, bias=True)

        self.conv_right1 = GCNConv(self.infeat, self.nhid, bias=True)
        self.conv_right2 = GCNConv(self.nhid * 2, self.nhid2, bias=True)
        self.conv_right3 = GCNConv(self.nhid2, self.outfeat, bias=True)

        self.conv_bipartite1 = GCNConv(self.infeat, self.nhid, bias=True)
        self.conv_bipartite2 = GCNConv(self.nhid, self.outfeat, bias=True)

        self.fc = nn.Linear(96, 64)
        # self.fc = nn.Linear(224, 64)
        self.bn = nn.BatchNorm1d(64)
        self.dropout_layer = nn.Dropout(0.3)

    def forward(self, batch, data):
        left_x = data.left_x
        left_edge_index = data.left_edge_index
        left_edge_attr = data.left_edge_attr
        right_x = data.right_x
        right_edge_index = data.right_edge_index
        right_edge_attr = data.right_edge_attr
        bipartite_x = data.bipartite_x
        bipartite_edge_index = data.bipartite_edge_index
        bipartite_edge_attr = data.bipartite_edge_attr

        # 1
        left_x = F.relu(self.conv_left1(left_x, left_edge_index, left_edge_attr))
        left_x = F.dropout(left_x, self.dropout, training=self.training)
        right_x = F.relu(self.conv_right1(right_x, right_edge_index, right_edge_attr))
        right_x = F.dropout(right_x, self.dropout, training=self.training)

        bipartite_x = F.relu(self.conv_bipartite1(bipartite_x, bipartite_edge_index, bipartite_edge_attr))
        bipartite_x_readout1 = torch.cat([gmp(bipartite_x, batch), gap(bipartite_x, batch)], dim=1)
        bipartite_x = F.dropout(bipartite_x, self.dropout, training=self.training)

        left_x_bi, right_x_bi = split_bipartite_att(bipartite_x)

        # 半脑内一致性
        left_consistency1 = F.pairwise_distance(left_x_bi, left_x, p=2)
        right_consistency1 = F.pairwise_distance(right_x_bi, right_x, p=2)

        right_x = torch.cat([right_x, right_x_bi], dim=1)
        left_x = torch.cat([left_x, left_x_bi], dim=1)

        # 2
        left_x = F.relu(self.conv_left2(left_x, left_edge_index, left_edge_attr))
        # left_x = F.dropout(left_x, self.dropout, training=self.training)
        right_x = F.relu(self.conv_right2(right_x, right_edge_index, right_edge_attr))
        # right_x = F.dropout(right_x, self.dropout, training=self.training)

        # 3
        bipartite_x = torch.cat([left_x, right_x], dim=1)
        bipartite_x = self.conv_bipartite2(bipartite_x, bipartite_edge_index, bipartite_edge_attr)
        # bipartite_x = F.relu(bipartite_x)
        bipartite_x_readout2 = torch.cat([gmp(bipartite_x, batch), gap(bipartite_x, batch)], dim=1)

        left_x_bi2, right_x_bi2 = split_bipartite_att(bipartite_x)

        left_x = self.conv_left3(left_x, left_edge_index, left_edge_attr)
        left_x_readout1 = torch.cat([gmp(left_x, batch), gap(left_x, batch)], dim=1)
        # left_x = F.dropout(left_x, self.dropout, training=self.training)
        right_x = self.conv_right3(right_x, right_edge_index, right_edge_attr)
        right_x_readout1 = torch.cat([gmp(right_x, batch), gap(right_x, batch)], dim=1)
        # right_x = F.dropout(right_x, self.dropout, training=self.training)

        # 半脑内一致性2
        left_consistency2 = F.pairwise_distance(left_x_bi2, left_x, p=2)
        right_consistency2 = F.pairwise_distance(right_x_bi2, right_x, p=2)

        features = torch.cat([bipartite_x_readout2, left_x_readout1, right_x_readout1], dim=1)
        # features = torch.cat([bipartite_x_readout1, bipartite_x_readout2], dim=1)

        features = self.dropout_layer(self.bn(F.relu(self.fc(features))))

        return features, bipartite_x_readout1, bipartite_x_readout2, left_consistency1, right_consistency1, left_consistency2, right_consistency2


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        # self.cosineclassfier = CosineClassfier(hidden_size2, num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        logits = self.fc3(x)
        # x = self.cosineclassfier(x)
        return logits, x


def split_bipartite_att(bipartite_att):
    left_torch = torch.zeros_like(bipartite_att)
    right_torch = torch.zeros_like(bipartite_att)

    left_torch[::2, ::2] = bipartite_att[::2, ::2]
    right_torch[1::2, 1::2] = bipartite_att[1::2, 1::2]

    return left_torch, right_torch

