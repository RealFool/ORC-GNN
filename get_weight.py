import os
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from utils.brainnetwork_reader import MyNetworkReader
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from model.gnn import MyGCN
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from utils.attacked_data import MyReadPtbData
from retrying import retry
import torch.nn as nn
import copy
import deepdish as dd
import Generate_UpperFeat_02 as generate_upperfeat
from model.BoundaryLearning import BoundaryLearning
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--Openset_pre', type=bool, default=True, help='Is open set recognition data preprocessing enabled')
parser.add_argument('--label_unseen', type=int, default=4, help='numofROI')
parser.add_argument('--nclass', type=int, default=4, help='num of classes')

parser.add_argument('--mat_dir', type=str, default=os.path.join(os.getcwd(), "data", "Brainnet"), help='BrainNet_dir')
parser.add_argument('--upper_dir', type=str, default=os.path.join(os.getcwd(), "data", "UpperFeat_HC_MDD"), help='Upper_feature_dir')

parser.add_argument('--lam_group', type=float, default=0.05, help='lam_group')
parser.add_argument('--CV_splits', type=int, default=5, help='CV_splits')
parser.add_argument('--val_slice', type=float, default=0.8, help='val_slice')
parser.add_argument('--train_with_val', type=bool, default=False, help='If performing val')
parser.add_argument('--numofROI', type=int, default=116, help='numofROI')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batchSize', type=int, default=200, help='batche size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler')
parser.add_argument('--weightdecay', type=float, default=5e-4, help='regularization')
parser.add_argument('--infeat', type=int, default=116, help='in_feature_dim of GCN')
parser.add_argument('--outfeat', type=int, default=32, help='out_feature_dim of GCN')
parser.add_argument('--attack', type=str, default='no', choices=['no', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=10, help="noise ptb_rate")
parser.add_argument('--dataset', type=str, default="MultiBrain", help='dataset')
parser.add_argument('--land_layers', type=int, default=8, help='land')
parser.add_argument('--lasland', type=int, default=32, help='lasland')
parser.add_argument('--resolution', type=int, default=1000, help='Resolution for landscapes')
parser.add_argument('--lamb_1', type=float, default=0.4, help='Consistency LOSS')
parser.add_argument('--lamb_2', type=float, default=0.001, help='L2 regularization')
# parser.add_argument('--lamb_1', type=float, default=0.5, help='Consistency LOSS')
# parser.add_argument('--lamb_2', type=float, default=0, help='L2 regularization')
parser.add_argument('--ablation', type=bool, default=False, help='L2 regularization')
opt = parser.parse_args()

# save_model_path = './save_model'
save_model_path = './setting/save_model'
# save_model_path = './compare/save_model'
seen_class_group = ''
if opt.Openset_pre:
    # 未知类
    # 20% ['ABIDE', 'BD', 'SCHZ']
    seen_class_group_1 = ['HC_MDD_BD_SCHZ', 'HC_MDD_SCHZ_ABIDE', 'HC_MDD_BD_ABIDE']
    # 40% ['ABIDE_MDD', 'BD_MDD', 'SCHZ_MDD']
    seen_class_group_2 = ['HC_BD_SCHZ', 'HC_SCHZ_ABIDE', 'HC_BD_ABIDE']
    # 60% ['ABIDE_MDD_SCHZ', 'ABIDE_MDD_BD', 'BD_SCHZ_ABIDE']
    seen_class_group_3 = ['HC_BD', 'HC_SCHZ', 'HC_MDD']
    seen_class_group = seen_class_group_1[0]
    # seen_class_group = seen_class_group_2[0]
    # seen_class_group = seen_class_group_3[2]

    if seen_class_group != 'HC_MDD_BD_SCHZ':
        opt.upper_dir = opt.upper_dir + '_' + seen_class_group
        opt.mat_dir = opt.mat_dir + '_' + seen_class_group

    # 指定保存模型的路径
    # save_model_path = save_model_path + '_' + seen_class_group
    save_model_path = os.path.join(save_model_path, seen_class_group, str(opt.lamb_1) + '_' + str(opt.lamb_2))
    if opt.ablation:
        save_model_path = os.path.join(save_model_path, 'ablation')

    opt.nclass = seen_class_group.count("_") + 1
    opt.label_unseen = opt.nclass

#################### Parameter Initialization #######################
lam_group = opt.lam_group
val_slice = opt.val_slice
land = opt.land_layers
resolution = opt.resolution
CV_splits = opt.CV_splits
numofROI = opt.numofROI
lasland = opt.lasland
train_with_val = opt.train_with_val
loss_min = 1e5
upper_dir = opt.upper_dir
BrainNetwork_dir = os.path.join(opt.mat_dir, str(lam_group))
num_epoch = opt.n_epochs
model_parameters_shw = False


def harf_matrix(zeros_on_odd=False, extra_padding=0):
    a = int((116 - extra_padding) / 2)
    A = np.ones((a, a))
    rows, cols = A.shape
    # Calculate the size of the new matrix
    new_rows = rows * 2 + extra_padding
    new_cols = cols * 2 + extra_padding

    # Create a new matrix filled with zeros
    expanded = np.zeros((new_rows, new_cols), dtype=A.dtype)

    if zeros_on_odd:
        # Place the original elements in the even rows and columns of the new matrix
        expanded[1:rows * 2:2, 1:cols * 2:2] = A
    else:
        # Place the original elements in the odd rows and columns of the new matrix
        expanded[:rows * 2:2, :cols * 2:2] = A

    return expanded


def bipartite_matrix(left_A_binary, right_A_binary, extra_padding=0):
    a = int(116 - extra_padding)
    A = np.ones((a, a))
    rows, cols = A.shape
    # Calculate the size of the new matrix
    new_rows = rows + extra_padding
    new_cols = cols + extra_padding

    # Create a new matrix filled with zeros
    expanded = np.zeros((new_rows, new_cols), dtype=A.dtype)

    expanded[:rows, :cols] = A
    bipartite_A_binary = expanded - left_A_binary - right_A_binary
    return bipartite_A_binary

def full_matrix(extra_padding=0):
    a = int(116 - extra_padding)
    A = np.ones((a, a))
    rows, cols = A.shape
    # Calculate the size of the new matrix
    new_rows = rows + extra_padding
    new_cols = cols + extra_padding

    # Create a new matrix filled with zeros
    expanded = np.zeros((new_rows, new_cols), dtype=A.dtype)

    expanded[:rows, :cols] = A

    return expanded


def fusion_matrix(A_binary, weight):
    # 计算每个原始特征的总体重要性
    C = np.sum(np.abs(weight), axis=0)  # 沿着特征转换方向累加
    # 计算 C 的外积，生成加权系数矩阵
    C_outer = np.outer(C, C)
    # 生成加权的邻接矩阵 A'(F)
    fusion_weight = A_binary * C_outer  # 使用元素乘积对 A(F) 进行加权
    return fusion_weight


def roi_contribution(fusion_matrix, top_n=10):
    # 假设您的文件路径，文件名应该替换成您的实际文件路径
    file_name = './aal.txt'

    # 读取文件并提取第二列（节点名称）
    names = []
    with open(file_name, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  # 假设每行的格式为value1,value2,value3
            names.append(parts[1])  # 获取第二列数据

    # 转换为NumPy数组以便索引
    names = np.array(names)

    # 计算每个节点的强度（即每行的总和）
    node_strength = np.sum(fusion_matrix, axis=1)  # 对行求和得到每个节点的出强度

    # 获取强度从大到小的节点索引
    sorted_indices = np.argsort(node_strength)[::-1]

    # 假设 names 包含节点名称，每个节点一个名称
    # names = ['Node ' + str(i + 1) for i in range(116)]  # 示例数据，实际使用时应替换成您的节点名称

    # 打印排名靠前的ROI，例如，前10个
    # top_n = 10
    for i in range(top_n):
        idx = sorted_indices[i]
        strength = node_strength[idx]
        print(f"ROI #{i + 1}: 节点 {idx} :{names[idx]}, 强度: {strength}")


# def edge_contribution(fusion_matrix, top_n=10):
#     # 假设您的文件路径，文件名应该替换成您的实际文件路径
#     file_name = './aal.txt'
#
#     # 读取文件并提取第二列（节点名称）
#     names = []
#     with open(file_name, 'r') as file:
#         for line in file:
#             parts = line.strip().split(',')  # 假设每行的格式为value1,value2,value3
#             names.append(parts[1])  # 获取第二列数据
#
#     # 转换为NumPy数组以便索引
#     names = np.array(names)
#
#     # 将矩阵展平
#     flat_adj = fusion_matrix.flatten()
#
#     # 获取展平矩阵值从大到小的索引
#     sorted_indices = np.argsort(flat_adj)[::-1]
#
#     # 将一维索引转换回二维矩阵的坐标
#     rows, cols = np.divmod(sorted_indices, fusion_matrix.shape[1])
#
#     # 提取排名靠前的连接，例如，前10个
#     # top_n = 50
#     for i in range(top_n):
#         row = rows[i]
#         col = cols[i]
#         # 获取并打印节点名称及其连接权重
#         name_row = names[row]
#         name_col = names[col]
#         weight = fusion_matrix[row, col]
#         # print(f"连接 #{i + 1}: 节点 {row} 到 节点 {col}, 权重: {weight}")
#         print(f"连接 #{i + 1}: 节点 {row}({name_row}) 到 节点 {col}({name_col}), 权重: {weight}")


def edge_contribution(fusion_matrix, top_n=10):
    # 假设您的文件路径，文件名应该替换成您的实际文件路径
    file_name = './aal.txt'

    # 读取文件并提取第二列（节点名称）
    names = []
    with open(file_name, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  # 假设每行的格式为value1,value2,value3
            names.append(parts[1])  # 获取第二列数据

    # 转换为NumPy数组以便索引
    names = np.array(names)

    # 获取矩阵的上三角部分的索引
    triu_indices = np.triu_indices(fusion_matrix.shape[0], k=1)

    # 获取上三角部分的连接权重
    triu_weights = fusion_matrix[triu_indices]

    # 获取按权重排序后的连接索引
    sorted_indices = np.argsort(triu_weights)[::-1]

    # 获取排名靠前的连接，例如，前10个
    for i in range(top_n):
        index = sorted_indices[i]
        row, col = triu_indices[0][index], triu_indices[1][index]
        # 获取并打印节点名称及其连接权重
        name_row, name_col = names[row], names[col]
        weight = triu_weights[index]
        print(f"连接 #{i + 1}: 节点 {row}({name_row}) 到 节点 {col}({name_col}), 权重: {weight}")


def eage_roi_edge_contribution(model):
    full_A_binary = full_matrix(extra_padding=8)
    left_A_binary = harf_matrix(zeros_on_odd=False, extra_padding=8)
    right_A_binary = harf_matrix(zeros_on_odd=True, extra_padding=8)
    bipartite_A_binary = bipartite_matrix(left_A_binary, right_A_binary, extra_padding=8)

    full_brain_weight1 = model.conv1.weight.detach().cpu().numpy()
    full_brain_weight2 = model.conv2.weight.detach().cpu().numpy()

    left_brain_weight1 = model.harf_bipartite_network.conv_left1.weight.detach().cpu().numpy()
    # left_brain_weight2 = model.harf_bipartite_network.conv_left2.weight.detach().cpu().numpy()
    left_brain_weight3 = model.harf_bipartite_network.conv_left3.weight.detach().cpu().numpy()

    right_brain_weight1 = model.harf_bipartite_network.conv_right1.weight.detach().cpu().numpy()
    # right_brain_weight2 = model.harf_bipartite_network.conv_right2.weight.detach().cpu().numpy()
    right_brain_weight3 = model.harf_bipartite_network.conv_right3.weight.detach().cpu().numpy()

    bipartite_brain_weight1 = model.harf_bipartite_network.conv_bipartite1.weight.detach().cpu().numpy()
    bipartite_brain_weight2 = model.harf_bipartite_network.conv_bipartite2.weight.detach().cpu().numpy()

    # 权重融合
    full_weight_combined = np.dot(full_brain_weight2, full_brain_weight1)  # [16, 116]
    harf_brain_weight2 = np.concatenate((left_brain_weight3, right_brain_weight3), axis=1)
    left_weight_combined = np.dot(harf_brain_weight2, left_brain_weight1)  # [16, 116]
    right_weight_combined = np.dot(harf_brain_weight2, right_brain_weight1)  # [16, 116]
    bipartite_weight_combined = np.dot(bipartite_brain_weight2, bipartite_brain_weight1)  # [16, 116]

    # 结构约束
    fusion_weight_full = fusion_matrix(full_A_binary, full_weight_combined)
    fusion_weight_left = fusion_matrix(left_A_binary, left_weight_combined)
    fusion_weight_right = fusion_matrix(right_A_binary, right_weight_combined)
    fusion_weight_bipartite = fusion_matrix(bipartite_A_binary, bipartite_weight_combined)

    # 差异矩阵
    # D = np.abs(fusion_weight_full - fusion_weight_left - fusion_weight_right)
    D = fusion_weight_full - fusion_weight_left - fusion_weight_right
    # D = D * bipartite_A_binary

    # 强弱连接融合矩阵
    weight_fusion = D + fusion_weight_bipartite

    # 将对角线元素置为0
    weight_fusion = weight_fusion - np.multiply(weight_fusion, np.eye(weight_fusion.shape[0]))

    return weight_fusion


def characteristic_eage_roi_edge_contribution(model, class_ids=1):
    # full_A_binary = full_matrix(extra_padding=8)
    left_A_binary = harf_matrix(zeros_on_odd=False, extra_padding=8)
    right_A_binary = harf_matrix(zeros_on_odd=True, extra_padding=8)
    bipartite_A_binary = bipartite_matrix(left_A_binary, right_A_binary, extra_padding=8)

    x_feat_weight = np.dot(model.classifier.weight.detach().cpu().numpy(),
                           model.mlp.fc2.weight.detach().cpu().numpy())

    x_feat_weight = np.dot(x_feat_weight, model.mlp.fc1.weight.detach().cpu().numpy())

    class_name = ['HC', 'MDD', 'BD', 'SCHZ']
    print("=============== ", class_name[class_ids], " ===============")
    class_feat_weight = x_feat_weight[class_ids]
    # full_brain_feat_weight = class_feat_weight[:160]  # [160]
    # full_brain_feat_weight = full_brain_feat_weight.reshape(1, full_brain_feat_weight.shape[0])  # [1,160]

    bipartite_brain_feat_weight = class_feat_weight[288:]  # [64]
    # [1, 64]
    bipartite_brain_feat_weight = bipartite_brain_feat_weight.reshape(1, bipartite_brain_feat_weight.shape[0])
    bipartite_brain_feat_weight = bipartite_brain_feat_weight * model.harf_bipartite_network.bn.weight.detach().cpu().numpy()
    bipartite_brain_feat_weight = np.dot(bipartite_brain_feat_weight,
                                         model.harf_bipartite_network.fc.weight.detach().cpu().numpy())  # [1, 96]

    bi_weight = bipartite_brain_feat_weight[:, :32]     # [1,32]
    # left_weight = bipartite_brain_feat_weight[:, 32:64]     # [1,32]
    # right_weight = bipartite_brain_feat_weight[:, 64:]     # [1,32]

    bi_weight = bi_weight.reshape(2, 16)  # [2,16]
    # 权重融合
    bipartite_brain_weight1 = model.harf_bipartite_network.conv_bipartite1.weight.detach().cpu().numpy()
    bipartite_brain_weight2 = model.harf_bipartite_network.conv_bipartite2.weight.detach().cpu().numpy()
    bi_weight = np.dot(bi_weight, bipartite_brain_weight2)
    bipartite_weight_combined = np.dot(bi_weight, bipartite_brain_weight1)

    # 结构约束
    fusion_weight_bipartite = fusion_matrix(bipartite_A_binary, bipartite_weight_combined)

    # 将对角线元素置为0
    weight_fusion = fusion_weight_bipartite - np.multiply(fusion_weight_bipartite, np.eye(fusion_weight_bipartite.shape[0]))

    return weight_fusion
    # 胼胝体
    # edge_contribution(fusion_weight_bipartite, top_n=20)
    # 感兴趣区
    # roi_contribution(fusion_weight_bipartite, top_n=10)


if __name__ == '__main__':
    # upperfea_len = 4005
    #
    # weight_fusion_mean = np.zeros((116, 116))
    #
    # class_name = ['HC', 'MDD', 'BD', 'SCHZ']
    # class_ids = 1
    #
    # for i in range(5):
    #
    #     model = MyGCN(opt.infeat, opt.outfeat, opt.nclass, numofROI, upperfea_len).to(
    #         device)  # infeat=116, outfeat=32, nclass=2
    #
    #     model_path = os.path.join(save_model_path, 'model_kf_' + str(i) + '.pth')
    #     model.load_state_dict(torch.load(model_path))
    #
    #     # for name, param in model.named_parameters():
    #     #     if 'weight' in name:  # 这将确保我们只打印出权重，不打印偏置等其他参数
    #     #         print(f"{name}: {param.size()}")
    #             # print(param.data)  # 或者使用 param.item() 如果它是一个标量
    #
    #     # 最大贡献边缘和ROI
    #     # weight_fusion = eage_roi_edge_contribution(model)
    #     weight_fusion = characteristic_eage_roi_edge_contribution(model, class_ids=class_ids)
    #
    #     weight_fusion_mean = weight_fusion_mean + weight_fusion
    #
    # weight_fusion_mean = np.abs(weight_fusion_mean/5)
    #
    # np.savetxt('weight_fusion.txt', weight_fusion_mean)
    # # np.savetxt(class_name[class_ids]+'_weight_fusion.txt', weight_fusion_mean)
    #
    # # 胼胝体
    # edge_contribution(weight_fusion_mean, top_n=10)
    # # 感兴趣区
    # # roi_contribution(weight_fusion, top_n=10)

    # weight_fusion_mean = np.loadtxt('./weight/weight_fusion.txt')
    # weight_fusion_mean = np.loadtxt('./weight/weight_fusion_bd.txt')
    # weight_fusion_mean = np.loadtxt('./weight/weight_fusion_mdd.txt')
    weight_fusion_mean = np.loadtxt('./weight/weight_fusion_schz.txt')

    # 胼胝体
    edge_contribution(weight_fusion_mean, top_n=10)

