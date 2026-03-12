# This code is developed based on Xiaoxiao Li, 2019/02/24
# The original version is at https://github.com/xxlya/BrainGNN_Pytorch/blob/main/imports/read_abide_stats_parall.py

import os.path as osp
from os import listdir
import os
import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_array
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
import scipy.io as scio
from nilearn import connectome
from tqdm import tqdm

BiGraph_Ratio = '0.1'
WoGraph_Ratio = '0.1'

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])
    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices


def split_new(data, batch, slices, type='whole_brain'):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])
    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    if type == 'whole_brain':
        edge_index_key = 'edge_index'
        x_key = 'x'
        edge_attr_key = 'edge_attr'
        y_key = 'y'
        pos_key = 'pos'
    elif type == 'left_brain':
        edge_index_key = 'left_edge_index'
        x_key = 'left_x'
        edge_attr_key = 'left_edge_attr'
        y_key = 'y'
        pos_key = 'left_pos'
    elif type == 'right_brain':
        edge_index_key = 'right_edge_index'
        x_key = 'right_x'
        edge_attr_key = 'right_edge_attr'
        y_key = 'y'
        pos_key = 'right_pos'
    else:
        edge_index_key = 'bipartite_edge_index'
        x_key = 'bipartite_x'
        edge_attr_key = 'bipartite_edge_attr'
        y_key = 'y'
        pos_key = 'bipartite_pos'

    slices[edge_index_key] = edge_slice
    if data.x is not None:
        slices[x_key] = node_slice
    if data.edge_attr is not None:
        slices[edge_attr_key] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices[y_key] = node_slice
        else:
            slices[y_key] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices[pos_key] = node_slice

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# def group_data_read(data_dir):
#     harfbrain_data_dir = os.path.join(data_dir, 'harfbrain', 'raw')
#     harfbrain_matfiles = [f for f in listdir(harfbrain_data_dir) if osp.isfile(osp.join(harfbrain_data_dir, f))]
#     harfbrain_matfiles.sort()
#
#     matfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
#     matfiles.sort()
#     Fun_dir = os.path.join(os.getcwd(), "data/Functional")
#     Funfiles = [m for m in listdir(Fun_dir) if osp.isfile(osp.join(Fun_dir, m))]
#     Funfiles.sort()
#
#     batch = []
#     pseudo = []
#     y_list = []
#     temp = []
#     subject = []
#     upper_triangle_feature = []
#     edge_att_list, edge_index_list, att_list, \
#         left_brain_edge_att_list, left_brain_edge_index_list, \
#         right_brain_edge_att_list, right_brain_edge_index_list, \
#         bipartite_edge_att_list, bipartite_edge_index_list, \
#         leftBrainAtt_list, rightBrainAtt_list, bipartiteBrainAtt_list, \
#         pseudo_left, pseudo_right, batch_left, batch_right, pseudo_bipartite, batch_bipartite = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
#     for network_name, timeseries_name, harfbrain_matfiles_name in tqdm(zip(matfiles, Funfiles, harfbrain_matfiles)):
#         res = individual_data_read(data_dir, network_name, Fun_dir, timeseries_name, harfbrain_data_dir,
#                                    harfbrain_matfiles_name)
#         temp.append(res)
#
#     for j in range(len(temp)):
#         edge_att_list.append(temp[j][0])
#         edge_index_list.append(temp[j][1] + j * temp[j][4])
#         att_list.append(temp[j][2])
#         y_list.append(temp[j][3])
#         batch.append([j] * temp[j][4])
#         pseudo.append(np.diag(np.ones(temp[j][4])))
#         subject.append(temp[j][5])
#         upper_triangle_feature.append(temp[j][6])
#         # Harf Brain
#         left_brain_edge_att_list.append(temp[j][7])
#         left_brain_edge_index_list.append(temp[j][8])
#         right_brain_edge_att_list.append(temp[j][9])
#         right_brain_edge_index_list.append(temp[j][10])
#         bipartite_edge_att_list.append(temp[j][11])
#         bipartite_edge_index_list.append(temp[j][12])
#         leftBrainAtt_list.append(temp[j][13])
#         rightBrainAtt_list.append(temp[j][14])
#         bipartiteBrainAtt_list.append(temp[j][15])
#
#         pseudo_left.append(np.diag(np.ones(temp[j][4])))
#         pseudo_right.append(np.diag(np.ones(temp[j][4])))
#         batch_left.append([j] * temp[j][4])
#         batch_right.append([j] * temp[j][4])
#         pseudo_bipartite.append(np.diag(np.ones(temp[j][4])))
#         batch_bipartite.append([j] * temp[j][4])
#
#     att_torch, edge_index_torch, y_torch, edge_att_torch, pseudo_torch, batch_torch = \
#         get_torch_graph_data(edge_att_list, edge_index_list, att_list, pseudo, y_list, batch)
#     # Harf Brain
#     left_att_torch, left_edge_index_torch, _, left_edge_att_torch, left_pseudo_torch, left_batch_torch = \
#         get_torch_graph_data(left_brain_edge_att_list, left_brain_edge_index_list, leftBrainAtt_list, pseudo_left,
#                              y_list, batch_left)
#     right_att_torch, right_edge_index_torch, _, right_edge_att_torch, right_pseudo_torch, right_batch_torch = \
#         get_torch_graph_data(right_brain_edge_att_list, right_brain_edge_index_list, rightBrainAtt_list, pseudo_right,
#                              y_list, batch_right)
#
#     bipartite_att_torch, bipartite_edge_index_torch, _, bipartite_edge_att_torch, bipartite_pseudo_torch, bipartite_batch_torch = \
#         get_torch_graph_data(bipartite_edge_att_list, bipartite_edge_index_list, bipartiteBrainAtt_list,
#                              pseudo_bipartite, y_list, batch_bipartite)
#
#     # data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos=pseudo_torch)
#     data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos=pseudo_torch,
#                 left_x=left_att_torch, left_edge_index=left_edge_index_torch, left_edge_attr=left_edge_att_torch, left_pos=left_pseudo_torch,
#                 right_x=right_att_torch, right_edge_index=right_edge_index_torch, right_edge_attr=right_edge_att_torch, right_pos=right_pseudo_torch,
#                 bipartite_x=bipartite_att_torch, bipartite_edge_index=bipartite_edge_index_torch, bipartite_edge_attr=bipartite_edge_att_torch, bipartite_pos=bipartite_pseudo_torch)
#
#     # data, slices = split(data, batch_torch)
#     slices = {}
#     data, slices = split_new(data, batch_torch, slices, type='whole_brain')
#     data, slices = split_new(data, left_batch_torch, slices, type='left_brain')
#     data, slices = split_new(data, right_batch_torch, slices, type='right_brain')
#     data, slices = split_new(data, bipartite_batch_torch, slices, type='bipartite_brain')
#
#     return data, slices, subject, upper_triangle_feature

def group_data_read(data_dir):
    harfbrain_data_dir = os.path.join(data_dir, 'harfbrain', 'raw')
    harfbrain_matfiles = [f for f in listdir(harfbrain_data_dir) if osp.isfile(osp.join(harfbrain_data_dir, f))]
    harfbrain_matfiles.sort()

    matfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    matfiles.sort()
    if "openset" in harfbrain_data_dir:
        # class_name_list = ['HC', 'MDD', 'BD', 'SCHZ', 'ABIDE', 'ningbo']
        class_name_list = ['HC', 'MDD', 'BD', 'SCHZ', 'ABIDE', 'ningbo76']
        class_name = ''
        for i in class_name_list:
            if i in harfbrain_data_dir:
                class_name = i
                break
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/openset", class_name)
        # 补充实验
        Fun_dir = os.path.join(os.getcwd(), "data/supplement2/Functional/openset", class_name)
    else:
        # 补充实验，HC,MDD,ABIDE多样本，已知类HC_MDD
        # Fun_dir = os.path.join(os.getcwd(), "data/supplement3/Functional/closeset_HC_MDD")
        # 补充实验，HC,MDD,ABIDE多样本，已知类HC_ABIDE
        # Fun_dir = os.path.join(os.getcwd(), "data/supplement3/Functional/closeset_HC_ABIDE")
        # 补充实验，HC,MDD,ABIDE多样本，已知类HC_MDD_ABIDE
        Fun_dir = os.path.join(os.getcwd(), "data/supplement3/Functional/closeset_HC_MDD_ABIDE")

        # 已知类：HC_MDD_BD_SCHZ
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/closeset")
        # 已知类：HC\BD\SCHZ
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/closeset_HC_BD_SCHZ")
        # 已知类：HC\MDD\BD
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/closeset_HC_MDD_BD")
        # 已知类：HC\MDD
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/closeset_HC_MDD")
        # 已知类：HC\BD
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/closeset_HC_BD")

        # 已知类：HC_MDD_BD_ABIDE
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/closeset_HC_MDD_BD_ABIDE")
        # 已知类：HC_MDD_SCHZ_ABIDE
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/closeset_HC_MDD_SCHZ_ABIDE")
        # 已知类：HC_SCHZ_ABIDE
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/closeset_HC_SCHZ_ABIDE")
        # 已知类：HC_BD_ABIDE
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/closeset_HC_BD_ABIDE")
        # 已知类：HC_SCHZ
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/closeset_HC_SCHZ")
        # 已知类：HC_ABIDE
        # Fun_dir = os.path.join(os.getcwd(), "data/Functional/closeset_HC_ABIDE")

    Funfiles = [m for m in listdir(Fun_dir) if osp.isfile(osp.join(Fun_dir, m))]
    Funfiles.sort()

    temp = []
    subject = []
    upper_triangle_feature = []
    for network_name, timeseries_name, harfbrain_matfiles_name in tqdm(zip(matfiles, Funfiles, harfbrain_matfiles)):
        res = individual_data_read(data_dir, network_name, Fun_dir, timeseries_name, harfbrain_data_dir,
                                   harfbrain_matfiles_name)
        temp.append(res)

    data_list = []
    for j in range(len(temp)):
        subject.append(temp[j][5])
        upper_triangle_feature.append(temp[j][6])
        edge_attr = torch.from_numpy(temp[j][0]).float()
        edge_index = torch.from_numpy(temp[j][1]).long()
        x = torch.from_numpy(temp[j][2]).float()
        y = torch.from_numpy(temp[j][3]).long()
        pos = torch.from_numpy(np.diag(np.ones(temp[j][4]))).float()
        
        left_edge_attr = torch.from_numpy(temp[j][7]).float()
        left_edge_index = torch.from_numpy(temp[j][8]).long()
        right_edge_attr = torch.from_numpy(temp[j][9]).float()
        right_edge_index = torch.from_numpy(temp[j][10]).long()
        bipartite_edge_attr = torch.from_numpy(temp[j][11]).float()
        bipartite_edge_index = torch.from_numpy(temp[j][12]).long()
        
        left_x = torch.nan_to_num(torch.from_numpy(temp[j][13]).float())
        right_x = torch.nan_to_num(torch.from_numpy(temp[j][14]).float())
        bipartite_x = torch.nan_to_num(torch.from_numpy(temp[j][15]).float())
        
        left_pos = torch.from_numpy(np.diag(np.ones(temp[j][4]))).float()
        right_pos = torch.from_numpy(np.diag(np.ones(temp[j][4]))).float()
        bipartite_pos = torch.from_numpy(np.diag(np.ones(temp[j][4]))).float()
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            left_x=left_x,
            left_edge_index=left_edge_index,
            left_edge_attr=left_edge_attr,
            left_pos=left_pos,
            right_x=right_x,
            right_edge_index=right_edge_index,
            right_edge_attr=right_edge_attr,
            right_pos=right_pos,
            bipartite_x=bipartite_x,
            bipartite_edge_index=bipartite_edge_index,
            bipartite_edge_attr=bipartite_edge_attr,
            bipartite_pos=bipartite_pos
        )
        data_list.append(data)

    return data_list, subject, upper_triangle_feature

def get_torch_graph_data(edge_att_list, edge_index_list, att_list, pseudo, y_list, batch):
    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()

    return att_torch, edge_index_torch, y_torch, edge_att_torch, pseudo_torch, batch_torch


def individual_data_read(net_dir, netfile, series_dir, seriesfile, harfbrain_data_dir, harfbrain_matfiles_name,
                         variable_Fun='ROISignals',
                         variable_bn='Brainnetwork'):
    SR_brain = scio.loadmat(osp.join(net_dir, netfile))[variable_bn]

    # 构边比例消融
    # SR_brain = get_ratio_mat(SR_brain, net_dir, is_bi_brain=False)

    Fun_series = scio.loadmat(osp.join(series_dir, seriesfile))[variable_Fun]
    # if Fun_series.shape[1] == 116:
    #     # 去除小脑区域
    #     Fun_series = Fun_series[:, 0:90]
    # Sub_name_curr = netfile[:-13]
    # Sub_name_curr = netfile[:-10]
    # Sub_name_curr = netfile[:-18]
    Sub_name_curr = netfile[:-16]  # 'S20-1-0001_net_0.3_0.5.mat'
    conn_measure = connectome.ConnectivityMeasure(kind='partial correlation')
    connectivity = conn_measure.fit_transform([Fun_series])

    # 补充实验，已知HC_MDD
    Sub = os.path.join(os.getcwd(), "data", "supplement", "suppelment_hc_mdd_abide.csv")
    # 补充实验，已知HC_ABIDE
    # Sub = os.path.join(os.getcwd(), "data", "supplement", "suppelment_hc_abide_mdd.csv")

    # 已知类：HC_MDD_BD_SCHZ
    # Sub = os.path.join(os.getcwd(), "data", "ALLPhenotypic_blance_5class.csv")
    # 已知类：HC\BD\SCHZ
    # Sub = os.path.join(os.getcwd(), "data", "ALLPhenotypic_HC_BD_SCHZ.csv")
    # 已知类：HC\MDD\BD
    # Sub = os.path.join(os.getcwd(), "data", "ALLPhenotypic_HC_MDD_BD.csv")
    # 已知类：HC\MDD
    # Sub = os.path.join(os.getcwd(), "data", "ALLPhenotypic_HC_MDD.csv")
    # 已知类：HC\BD
    # Sub = os.path.join(os.getcwd(), "data", "ALLPhenotypic_HC_BD.csv")

    # 已知类：HC_MDD_BD_ABIDE
    # Sub = os.path.join(os.getcwd(), "data", "ALLPhenotypic_HC_MDD_BD_ABIDE.csv")
    # 已知类：HC_MDD_SCHZ_ABIDE
    # Sub = os.path.join(os.getcwd(), "data", "ALLPhenotypic_HC_MDD_SCHZ_ABIDE.csv")
    # 已知类：HC_SCHZ_ABIDE
    # Sub = os.path.join(os.getcwd(), "data", "ALLPhenotypic_HC_SCHZ_ABIDE.csv")
    # 已知类：HC_BD_ABIDE
    # Sub = os.path.join(os.getcwd(), "data", "ALLPhenotypic_HC_BD_ABIDE.csv")
    # 已知类：HC_SCHZ
    # Sub = os.path.join(os.getcwd(), "data", "ALLPhenotypic_HC_SCHZ.csv")
    # 已知类：HC_ABIDE
    # Sub = os.path.join(os.getcwd(), "data", "ALLPhenotypic_HC_ABIDE.csv")

    if not os.path.isfile(Sub):
        print(Sub + 'does not exist!')
    else:
        if Sub.endswith('.csv'):
            Sub_gro = np.genfromtxt(Sub, dtype=str,
                                    delimiter=',',
                                    skip_header=1,
                                    usecols=(0, 1, 2, 3))

    # 创建一个布尔索引数组，表示哪些行的第0列（idx）等于 Sub_name_curr
    matches = Sub_gro[:, 0] == Sub_name_curr
    # 使用布尔索引来筛选出满足条件的行，并选择这些行的1, 2, 3列
    filtered_rows = Sub_gro[matches][:, [1, 2, 3]]
    Sub_label = filtered_rows[0][0]
    Sub_age = filtered_rows[0][1]
    Sub_gender = filtered_rows[0][2]

    # Sub_label = [label for idx, label in Sub_gro if idx == Sub_name_curr]
    # from adj matrix to too matrix
    num_nodes = SR_brain.shape[0]
    G = from_numpy_array(SR_brain)
    A = nx.to_scipy_sparse_array(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = SR_brain[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)
    att = np.transpose(connectivity[0])
    label = np.array([int(x) for x in Sub_label])

    upper_triangle_feature = extract_upper_triangle(att, Sub_age, Sub_gender)

    left_brain_edge_att, left_brain_edge_index, \
        right_brain_edge_att, right_brain_edge_index, \
        bipartite_edge_att, bipartite_edge_index, \
        leftBrainAtt, rightBrainAtt, bipartiteBrainAtt = individual_harfbrain_data_read(harfbrain_data_dir,
                                                                                        harfbrain_matfiles_name, net_dir)

    return edge_att.data.numpy(), edge_index.data.numpy(), att, label, num_nodes, Sub_name_curr, upper_triangle_feature, \
        left_brain_edge_att, left_brain_edge_index, right_brain_edge_att, right_brain_edge_index, bipartite_edge_att, bipartite_edge_index, \
        leftBrainAtt, rightBrainAtt, bipartiteBrainAtt


def individual_harfbrain_data_read(harfbrain_data_dir, harfbrain_matfiles_name, net_dir):
    leftBrainnetwork = scio.loadmat(osp.join(harfbrain_data_dir, harfbrain_matfiles_name))['LeftBrainNetwork']
    rightBrainnetwork = scio.loadmat(osp.join(harfbrain_data_dir, harfbrain_matfiles_name))['RightBrainNetwork']
    bipartite_adjacency_matrix = scio.loadmat(osp.join(harfbrain_data_dir, harfbrain_matfiles_name))['BipartiteBrainNetwork']

    # 构边比例消融
    # leftBrainnetwork = get_ratio_mat(leftBrainnetwork, net_dir, is_bi_brain=True)
    # rightBrainnetwork = get_ratio_mat(rightBrainnetwork, net_dir, is_bi_brain=True)
    # bipartite_adjacency_matrix = get_ratio_mat(bipartite_adjacency_matrix, net_dir, is_bi_brain=True)

    # from adj matrix to too matrix
    left_brain_edge_att, left_brain_edge_index = get_edge_info(leftBrainnetwork)
    right_brain_edge_att, right_brain_edge_index = get_edge_info(rightBrainnetwork)
    bipartite_edge_att, bipartite_edge_index = get_edge_info(bipartite_adjacency_matrix)

    leftBrainAtt = scio.loadmat(osp.join(harfbrain_data_dir, harfbrain_matfiles_name))['LeftBrainAtt']
    rightBrainAtt = scio.loadmat(osp.join(harfbrain_data_dir, harfbrain_matfiles_name))['RightBrainAtt']
    bipartiteBrainAtt = scio.loadmat(osp.join(harfbrain_data_dir, harfbrain_matfiles_name))['BipartiteBrainAtt']

    return left_brain_edge_att, left_brain_edge_index, \
        right_brain_edge_att, right_brain_edge_index, \
        bipartite_edge_att, bipartite_edge_index, \
        leftBrainAtt, rightBrainAtt, bipartiteBrainAtt


def get_edge_info(brain_network):
    num_nodes = brain_network.shape[0]
    G = from_numpy_array(brain_network)
    A = nx.to_scipy_sparse_array(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = brain_network[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)

    return edge_att.data.numpy(), edge_index.data.numpy()


def extract_upper_triangle(matrix, Sub_age, Sub_gender):
    # 提取上三角部分，不包括对角线
    upper_triangle = matrix[np.triu_indices(90, k=1)]
    # upper_triangle = np.append(upper_triangle, float(Sub_age))
    # upper_triangle = np.append(upper_triangle, float(Sub_gender))
    return upper_triangle


def get_ratio_mat(brain_net, net_dir, is_bi_brain=False):
    ratio_str = net_dir[-11:-4].split('_')
    brain_ratio = ratio_str[0]
    if is_bi_brain:
        brain_ratio = ratio_str[1]
    # 根据比例计算动态阈值
    all_values = brain_net.flatten()
    threshold_value = np.percentile(all_values, (1 - float(brain_ratio)) * 100)
    # 应用动态阈值过滤矩阵
    Brain_spa_mat = np.where(np.abs(brain_net) >= threshold_value, brain_net, 0)
    return Brain_spa_mat
