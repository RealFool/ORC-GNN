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
from model.gnn_ablation_bi_li import MyGCN
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
supplement_file = "supplement3"
parser.add_argument('--Openset_pre', type=bool, default=True, help='Is open set recognition data preprocessing enabled')
parser.add_argument('--label_unseen', type=int, default=2, help='numofROI')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')

# 补充实验,已知类HC_MDD and HC_ABIDE
# parser.add_argument('--mat_dir', type=str, default=os.path.join(os.getcwd(), "data", supplement_file, "Brainnet"), help='BrainNet_dir')
# parser.add_argument('--upper_dir', type=str, default=os.path.join(os.getcwd(), "data", supplement_file, "UpperFeat"), help='Upper_feature_dir')

parser.add_argument('--mat_dir', type=str, default=os.path.join(os.getcwd(), "data", "Brainnet"), help='BrainNet_dir')
parser.add_argument('--upper_dir', type=str, default=os.path.join(os.getcwd(), "data", "UpperFeat"), help='Upper_feature_dir')

parser.add_argument('--lam_group', type=float, default=0.05, help='lam_group')
parser.add_argument('--CV_splits', type=int, default=5, help='CV_splits')
parser.add_argument('--val_slice', type=float, default=0.8, help='val_slice')
parser.add_argument('--train_with_val', type=bool, default=False, help='If performing val')
parser.add_argument('--numofROI', type=int, default=116, help='numofROI')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batchSize', type=int, default=200, help='batche size')
# parser.add_argument('--batchSize', type=int, default=1500, help='batche size')  # 补充实验
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
# parser.add_argument('--lamb_1', type=float, default=0.4, help='Consistency LOSS')
# parser.add_argument('--lamb_2', type=float, default=0.001, help='L2 regularization')
parser.add_argument('--lamb_1', type=float, default=0.5, help='Consistency LOSS')
parser.add_argument('--lamb_2', type=float, default=0, help='L2 regularization')
parser.add_argument('--ablation', type=bool, default=False, help='L2 regularization')
opt = parser.parse_args()

# save_model_path = './save_model'
save_model_path = './setting/save_model'
# save_model_path = './compare/save_model'  # pre best
# save_model_path = './'+supplement_file+'/save_model'
seen_class_group = ''
supplement = False
if opt.Openset_pre:
    # 补充实验 未知类 ['ABIDE', 'MDD']
    # seen_class_group_sup = ['HC_MDD', 'HC_ABIDE', 'HC_MDD_ABIDE']

    # 未知类
    # 20% ['ABIDE', 'BD', 'SCHZ']
    seen_class_group_1 = ['HC_MDD_BD_SCHZ', 'HC_MDD_SCHZ_ABIDE', 'HC_MDD_BD_ABIDE']
    # 40% ['ABIDE_MDD', 'BD_MDD', 'SCHZ_MDD']
    seen_class_group_2 = ['HC_BD_SCHZ', 'HC_SCHZ_ABIDE', 'HC_BD_ABIDE']
    # 60% ['ABIDE_MDD_SCHZ', 'ABIDE_MDD_BD', 'BD_SCHZ_ABIDE']
    seen_class_group_3 = ['HC_BD', 'HC_SCHZ', 'HC_MDD']
    seen_class_group = seen_class_group_1[0]
    # seen_class_group = seen_class_group_2[2]
    # seen_class_group = seen_class_group_3[2]

    # seen_class_group = seen_class_group_sup[0]

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
# BrainNetwork_dir = os.path.join(opt.mat_dir, '0.3_0.5')  # 补充实验
num_epoch = opt.n_epochs
model_parameters_shw = False


# 已知类
def get_outputs(pred_test, features, centroids, best_delta=None, open=True):
    euc_dis = torch.norm(features - centroids[pred_test], 2, 1).view(-1)
    if open is True:
        # openset
        pred_test[euc_dis > 1 * best_delta[pred_test]] = opt.label_unseen
    return pred_test


def test_begin(loader, External_fea, state):
    model.eval()
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        loss_all = 0
        AUC = 0
        Sensitivity = 0
        Specificity = 0
        seq_idx = 0
        test_correct = []
        lebels = None
        for data in loader:
            lebels = data.y
            data = data.to(device)
            upper_fea = External_fea[seq_idx:data.num_graphs + seq_idx]
            seq_idx = seq_idx + data.num_graphs
            output, consistency_loss, harf_consistency_loss, x_feat, logits = model(data.x, data.edge_index, data.batch, data.edge_attr, upper_fea,
                                                     data)
            _, test_pred = torch.max(output, 1)
            test_correct.append((test_pred == data.y).sum().item())

    return loss_all / len(loader.dataset), sum(test_correct) / len(loader.dataset), sum(
        test_correct), Sensitivity, Specificity, AUC, x_feat, test_pred, lebels, output


def test_open_begin(unseen_loader, upperfea_unseen):
    AUC = 0
    Sensitivity = 0
    Specificity = 0
    F1 = 0
    test_correct = []
    model.eval()
    with torch.no_grad():
        labels_unseen = None
        test_preds_unseen = None
        features_unseen = None
        for unseen_data in unseen_loader:
            # labels_unseen = unseen_data.y
            labels_unseen = torch.full_like(unseen_data.y, opt.label_unseen)

            unseen_data = unseen_data.to(device)
            output, _, _, features_unseen, logits = model(unseen_data.x, unseen_data.edge_index, unseen_data.batch,
                                               unseen_data.edge_attr, upperfea_unseen, unseen_data)
            test_u_prob, test_preds_unseen = torch.max(output, 1)

            # 评估
            # test_correct.append((test_preds_unseen == data.y).sum().item())
            # cm = confusion_matrix(unseen_data.y.cpu().numpy(), test_preds_unseen.cpu().numpy())
            # # 计算敏感性和特异性
            # sensitivity = []
            # specificity = []
            # for i in range(opt.nclass):
            #     tp = cm[i, i]
            #     fn = sum(cm[i, :]) - tp
            #     fp = sum(cm[:, i]) - tp
            #     tn = cm.sum() - (tp + fn + fp)
            #
            #     sensitivity_i = tp / (tp + fn)
            #     specificity_i = tn / (tn + fp)
            #
            #     sensitivity.append(sensitivity_i)
            #     specificity.append(specificity_i)
            #
            # # 计算平均敏感性和特异性
            # Sensitivity = sum(sensitivity) / opt.nclass
            # Specificity = sum(specificity) / opt.nclass
            #
            # # 将标签进行二进制编码
            # y_true_binary = label_binarize(unseen_data.y.cpu().numpy(), classes=np.arange(opt.nclass))
            #
            # # 计算每个类别的AUC
            # auc_scores = []
            # for i in range(opt.nclass):
            #     auc_i = roc_auc_score(y_true_binary[:, i], output.txt[:, i].cpu().numpy())
            #     auc_scores.append(auc_i)
            # # 使用 Macro-Averaging 计算总体AUC
            # AUC = np.mean(auc_scores)
            #
            # # 计算F1得分
            # F1 = f1_score(unseen_data.y.cpu().numpy(), test_preds_unseen.cpu().numpy(), average='macro')

    # return features_unseen, preds_unseen, labels_unseen, sum(test_correct) / len(unseen_loader.dataset), sum(
    #     test_correct), Sensitivity, Specificity, AUC, test_pred, F1
    return features_unseen, test_preds_unseen, labels_unseen


def read_upperfeas(dir, te_idx):
    upperfeat_list = []
    External_Upperfeas = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    External_Upperfeas.sort()
    for idx, res in enumerate(External_Upperfeas):
        upperfeas_file = dd.io.load(os.path.join(dir, res))
        # read edge and edge attribute
        upperfeat = upperfeas_file['UpperFeat'][()]
        upperfeat_list.append(upperfeat)

    DevCurve_temp = torch.stack(upperfeat_list)
    upperfea_TE = torch.index_select(DevCurve_temp, 0, torch.LongTensor(te_idx))

    return upperfea_TE


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def get_unseen_loader(unseen_class):
    # unseen_dir = os.path.join(os.getcwd(), "data", supplement_file, "Brainnet", "openset", unseen_class, '0.3_0.5')  # 补充实验
    unseen_dir = os.path.join(os.getcwd(), "data", "Brainnet", "openset", unseen_class, str(lam_group))
    dataset_unseen = MyNetworkReader(unseen_dir)
    # 随机按已知测试集比例刷选未知类测试样本
    num_samples = len(dataset_unseen.indices())
    random.seed(123)

    selected_indices = random.sample(dataset_unseen.indices(), num_samples // CV_splits)
    # selected_indices = dataset_unseen.indices()  # 补充实验
    # if unseen_class == 'BD' or unseen_class == 'SCHZ' or unseen_class == 'ningbo' or unseen_class == 'ningbo76':
    #     selected_indices = dataset_unseen.indices()  # 补充实验
    # else:
    #     selected_indices = random.sample(dataset_unseen.indices(), 50)  # 补充实验

    unseen_dataset = dataset_unseen[torch.LongTensor(selected_indices).to(device)]
    unseen_loader = DataLoader(unseen_dataset, batch_size=opt.batchSize, shuffle=False)
    # unseen_upper_dir = os.path.join(os.getcwd(), "data", supplement_file, "UpperFeat", "openset", unseen_class)  # 补充实验
    unseen_upper_dir = os.path.join(os.getcwd(), "data", "UpperFeat", "openset", unseen_class)
    upperfeat_list = []
    External_Upperfeas = [f for f in os.listdir(unseen_upper_dir) if
                          os.path.isfile(os.path.join(unseen_upper_dir, f))]
    External_Upperfeas.sort()
    for idx, res in enumerate(External_Upperfeas):
        upperfeas_file = dd.io.load(os.path.join(unseen_upper_dir, res))
        # read edge and edge attribute
        upperfeat = upperfeas_file['UpperFeat'][()]
        upperfeat_list.append(upperfeat)

    DevCurve_temp = torch.stack(upperfeat_list)
    unseen_upperfea = torch.index_select(DevCurve_temp, 0, torch.LongTensor(selected_indices))

    return unseen_loader, unseen_upperfea


def makedir_check(path_to_check):
    # 提取目录部分
    directory = os.path.dirname(path_to_check)
    # 检查目录是否存在
    if not os.path.exists(directory):
        # 如果目录不存在，创建目录（包括所有必要的父目录）
        os.makedirs(directory)


def evaluate(preds, labels):
    correct = np.sum(preds == labels)
    total = preds.shape[0]
    acc = correct / total

    # ACC = accuracy_score(labels, preds)
    # recall_s = recall_score(labels, preds, average="weighted")
    # print('\033[1;36mAcc Open: %.5f\033[0m' % (accuracy_open))
    # 计算F1得分
    f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='macro')
    weighted_precision = precision_score(labels, preds, average='weighted')

    cm = confusion_matrix(labels, preds)
    # 提取TP, TN, FP, FN
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP) + TP
    # 计算每个类别的敏感性（召回率）和特异性
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # # 计算平均敏感性和特异性
    Sensitivity = sum(sensitivity) / (opt.nclass + 1)
    Specificity = sum(specificity) / (opt.nclass + 1)

    # 计算每个类别的样本数以用于加权
    class_counts = np.bincount(labels, minlength=opt.nclass+1)
    # 计算权重
    weights = class_counts / np.sum(class_counts)
    # 计算加权平均敏感性和特异性
    weighted_avg_sensitivity = np.sum(weights * sensitivity)
    weighted_avg_specificity = np.sum(weights * specificity)

    return acc, precision, Specificity, Sensitivity, f1, weighted_precision, weighted_avg_sensitivity, weighted_avg_specificity, weighted_f1

import csv

def append_list_to_csv(file_path, data_list):
    # 打开文件，使用'a'模式以追加方式写入
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入列表到CSV文件
        writer.writerow(data_list)

def append_list_to_csv_column(file_path, data_list):
    # 读取现有内容
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='') as csvfile:
            reader = list(csv.reader(csvfile))
    else:
        reader = []

    # 找到下一列的索引
    if reader:
        next_col_index = len(reader[0])
    else:
        next_col_index = 0

    # 确保CSV文件有足够的行来容纳新数据
    max_len = max(len(reader), len(data_list))
    for _ in range(max_len - len(reader)):
        reader.append([])

    # 将列表内容写入到下一列
    for row_index, value in enumerate(data_list):
        if row_index >= len(reader):
            reader.append([])
        if next_col_index >= len(reader[row_index]):
            reader[row_index].extend([''] * (next_col_index - len(reader[row_index]) + 1))
        reader[row_index][next_col_index] = value

    # 写入更新后的内容到CSV文件
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(reader)

######################### Define Dataloader ##########################
if __name__ == '__main__':

    unseen_loader = None
    unseen_upperfea = None
    if opt.Openset_pre:
        class_name = ['HC', 'MDD', 'BD', 'SCHZ', 'ABIDE', 'ningbo', 'ningbo76']
        # unseen 25%
        unseen_class1 = class_name[1]
        unseen_class2 = class_name[2]
        unseen_class3 = class_name[3]
        unseen_class4 = class_name[4]
        # unseen_class5 = class_name[5]   # 补充实验
        # unseen_class6 = class_name[6]   # 补充实验

        unseen_loader1, unseen_upperfea1 = get_unseen_loader(unseen_class1)

        unseen_loader2, unseen_upperfea2 = get_unseen_loader(unseen_class2)

        unseen_loader3, unseen_upperfea3 = get_unseen_loader(unseen_class3)

        unseen_loader4, unseen_upperfea4 = get_unseen_loader(unseen_class4)

        # unseen_loader5, unseen_upperfea5 = get_unseen_loader(unseen_class5)     # 补充实验
        # unseen_loader6, unseen_upperfea6 = get_unseen_loader(unseen_class6)  # 补充实验

    dataset = MyNetworkReader(BrainNetwork_dir)
    Sub_list = dataset.subject
    dataset._data.y = dataset._data.y.squeeze()
    skf = StratifiedKFold(n_splits=CV_splits, shuffle=True, random_state=0)
    valid_idx = None
    total_samples = 0
    total_acc = 0
    total_auc = 0
    total_sen = 0
    total_spe = 0
    total_f1 = 0
    Acc_list = []
    Prec_list = []
    Auc_list = []
    Sen_list = []
    Spe_list = []
    F1_list = []
    W_Prec_list = []
    W_Sen_list = []
    W_Spe_list = []
    W_F1_list = []

    k_ford_i = 0
    for _, test_idx in skf.split(dataset, dataset._data.y):
        test_dataset = dataset[torch.LongTensor(test_idx).to(device)]
        test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

        ############### External upper feature of individual-level subject ###############
        upperfea_TE = read_upperfeas(upper_dir, test_idx)

        ###############################  GCN Model instancing  #################################
        model = MyGCN(opt.infeat, opt.outfeat, opt.nclass, numofROI, upperfea_TE.shape[1]).to(
            device)  # infeat=116, outfeat=32, nclass=2

        model_path = os.path.join(save_model_path, 'model_kf_' + str(k_ford_i) + '.pth')
        model.load_state_dict(torch.load(model_path))
        loss, acc, corrects, sen, spe, auc, test_x_feat, test_pred, test_lebels, output_k = test_begin(test_loader, upperfea_TE,
                                                                                             state='test')
        print('~~~~~~~~~~~~~~~')
        print('test_accuracy: %.5f %%' % (100 * acc))

        model_dict_path = os.path.join(save_model_path, 'model_dict_kf_' + str(k_ford_i) + '.npy')
        loaded_model_dict = np.load(model_dict_path, allow_pickle=True).item()
        centroids = loaded_model_dict['centroids']
        x_feat_train = loaded_model_dict['x_feat_train']
        y_train = loaded_model_dict['y_train']

        ############################ 边界学习 ############################
        delta = None
        if opt.Openset_pre:
            print("################ Boundry Learning Begin ################")
            boundary_learning = BoundaryLearning(opt.nclass, device)
            # boundary_learning = ADB_BoundaryLearning(opt.nclass, device)
            optimizer_boundary = torch.optim.Adam(boundary_learning.parameters(), lr=0.01)
            density_same_class_all, density_different_class_all = boundary_learning.get_density(x_feat_train, y_train)
            boundary_loss_list = []
            delta_list = []
            for epoch in tqdm(range(1000)):
                optimizer_boundary.zero_grad()
                boundary_loss, delta = boundary_learning(torch.tensor(x_feat_train).to('cuda'), centroids, torch.tensor(y_train).to('cuda'),
                                                         density_same_class_all, density_different_class_all,
                                                         epoch)
                # boundary_loss, delta = boundary_learning(x_feat_train.detach(), centroids, y_train)
                boundary_loss.backward()
                optimizer_boundary.step()

                boundary_loss_list.append(boundary_loss.detach().cpu().numpy())
                delta_list.append(delta.detach().cpu().numpy())
            boundary_dict = {'centroids': centroids, 'delta': delta}
            np.save('./boundary_loss', np.array(boundary_loss_list))
            np.save('./delta_list', np.array(delta_list))
            boundary_path = os.path.join(save_model_path, 'boundary', 'boundary_kf_' + str(k_ford_i) + '.npy')
            makedir_check(boundary_path)
            np.save(boundary_path, boundary_dict)

        k_ford_i += 1

        ############################  Openset Testing  ##############################
        # 原始测试集样本和未知类样本合并为新的测试集，进行开集识别测试
        if opt.Openset_pre:
            # ['HC', 'MDD', 'BD', 'SCHZ', 'ABIDE', 'ningbo']
            features_unseen1, preds_unseen1, labels_unseen1 = test_open_begin(unseen_loader1, unseen_upperfea1)
            features_unseen2, preds_unseen2, labels_unseen2 = test_open_begin(unseen_loader2, unseen_upperfea2)
            features_unseen3, preds_unseen3, labels_unseen3 = test_open_begin(unseen_loader3, unseen_upperfea3)
            features_unseen4, preds_unseen4, labels_unseen4 = test_open_begin(unseen_loader4, unseen_upperfea4)
            # features_unseen5, preds_unseen5, labels_unseen5 = test_open_begin(unseen_loader5, unseen_upperfea5)    # 补充实验
            # features_unseen6, preds_unseen6, labels_unseen6 = test_open_begin(unseen_loader6, unseen_upperfea6)  # 补充实验

            features_seen = test_x_feat.detach()
            preds_seen = test_pred
            labels_seen = test_lebels

            if seen_class_group == 'HC_MDD_BD_SCHZ':
                # 4
                cat_features = torch.cat([features_seen, features_unseen4])
                cat_preds = torch.cat([preds_seen, preds_unseen4])
                labels = torch.cat([labels_seen, labels_unseen4])
            elif seen_class_group == 'HC_MDD_SCHZ_ABIDE':
                # 2
                cat_features = torch.cat([features_seen, features_unseen2])
                cat_preds = torch.cat([preds_seen, preds_unseen2])
                labels = torch.cat([labels_seen, labels_unseen2])
            elif seen_class_group == 'HC_MDD_BD_ABIDE':
                # 3
                cat_features = torch.cat([features_seen, features_unseen3])
                cat_preds = torch.cat([preds_seen, preds_unseen3])
                labels = torch.cat([labels_seen, labels_unseen3])
            elif seen_class_group == 'HC_BD_SCHZ':
                # 14
                cat_features = torch.cat([features_seen, features_unseen4, features_unseen1])
                cat_preds = torch.cat([preds_seen, preds_unseen4, preds_unseen1])
                labels = torch.cat([labels_seen, labels_unseen4, labels_unseen1])
            elif seen_class_group == 'HC_SCHZ_ABIDE':
                # 12
                cat_features = torch.cat([features_seen, features_unseen2, features_unseen1])
                cat_preds = torch.cat([preds_seen, preds_unseen2, preds_unseen1])
                labels = torch.cat([labels_seen, labels_unseen2, labels_unseen1])
            elif seen_class_group == 'HC_BD_ABIDE':
                # 13
                cat_features = torch.cat([features_seen, features_unseen3, features_unseen1])
                cat_preds = torch.cat([preds_seen, preds_unseen3, preds_unseen1])
                labels = torch.cat([labels_seen, labels_unseen3, labels_unseen1])
            elif seen_class_group == 'HC_BD':
                # 134
                cat_features = torch.cat([features_seen, features_unseen4, features_unseen1, features_unseen3])
                cat_preds = torch.cat([preds_seen, preds_unseen4, preds_unseen1, preds_unseen3])
                labels = torch.cat([labels_seen, labels_unseen4, labels_unseen1, labels_unseen3])
            elif seen_class_group == 'HC_SCHZ':
                # 124
                cat_features = torch.cat([features_seen, features_unseen4, features_unseen1, features_unseen2])
                cat_preds = torch.cat([preds_seen, preds_unseen4, preds_unseen1, preds_unseen2])
                labels = torch.cat([labels_seen, labels_unseen4, labels_unseen1, labels_unseen2])
            elif seen_class_group == 'HC_MDD' and supplement is False:    # 补充实验
                # 234
                cat_features = torch.cat([features_seen, features_unseen3, features_unseen4, features_unseen2])
                cat_preds = torch.cat([preds_seen, preds_unseen3, preds_unseen4, preds_unseen2])
                labels = torch.cat([labels_seen, labels_unseen3, labels_unseen4, labels_unseen2])
            elif supplement:  # 补充实验
                if seen_class_group == 'HC_MDD':
                    # 4 ABIDE
                    # cat_features = torch.cat([features_seen, features_unseen4])
                    # cat_preds = torch.cat([preds_seen, preds_unseen4])
                    # labels = torch.cat([labels_seen, labels_unseen4])

                    # 2 BD
                    # cat_features = torch.cat([features_seen, features_unseen2])
                    # cat_preds = torch.cat([preds_seen, preds_unseen2])
                    # labels = torch.cat([labels_seen, labels_unseen2])

                    # 3 SCHZ
                    # cat_features = torch.cat([features_seen, features_unseen3])
                    # cat_preds = torch.cat([preds_seen, preds_unseen3])
                    # labels = torch.cat([labels_seen, labels_unseen3])

                    # unseen_preds = get_outputs(preds_unseen3, features_unseen3, centroids, delta)
                    # preds = torch.cat([preds_seen, unseen_preds])

                    # 23 BD SCHZ
                    # cat_features = torch.cat([features_seen, features_unseen2, features_unseen3])
                    # cat_preds = torch.cat([preds_seen, preds_unseen2, preds_unseen3])
                    # labels = torch.cat([labels_seen, labels_unseen2, labels_unseen3])

                    # unseen_preds = get_outputs(torch.cat([preds_unseen2, preds_unseen3]), torch.cat([features_unseen2, features_unseen3]), centroids, delta)
                    # preds = torch.cat([preds_seen, unseen_preds])

                    # 234 BD SCHZ ABIDE
                    cat_features = torch.cat([features_seen, features_unseen2, features_unseen3, features_unseen4])
                    cat_preds = torch.cat([preds_seen, preds_unseen2, preds_unseen3, preds_unseen4])
                    labels = torch.cat([labels_seen, labels_unseen2, labels_unseen3, labels_unseen4])

                    # unseen_preds = get_outputs(torch.cat([preds_unseen2, preds_unseen3, preds_unseen4]), torch.cat([features_unseen2, features_unseen3, features_unseen4]), centroids, delta)
                    # preds = torch.cat([preds_seen, unseen_preds])

                    # 5 ningbo
                    # cat_features = torch.cat([features_seen, features_unseen5])
                    # cat_preds = torch.cat([preds_seen, preds_unseen5])
                    # labels = torch.cat([labels_seen, labels_unseen5])

                    # 5 ningbo76
                    # cat_features = torch.cat([features_seen, features_unseen6])
                    # cat_preds = torch.cat([preds_seen, preds_unseen6])
                    # labels = torch.cat([labels_seen, labels_unseen6])

                elif seen_class_group == 'HC_ABIDE':
                    # 1 MDD
                    cat_features = torch.cat([features_seen, features_unseen1])
                    cat_preds = torch.cat([preds_seen, preds_unseen1])
                    labels = torch.cat([labels_seen, labels_unseen1])
                    # 12 MDD, BD
                    # cat_features = torch.cat([features_seen, features_unseen1, features_unseen2])
                    # cat_preds = torch.cat([preds_seen, preds_unseen1, preds_unseen2])
                    # labels = torch.cat([labels_seen, labels_unseen1, labels_unseen2])
                    # 123 MDD, BD, SCHZ
                    # cat_features = torch.cat([features_seen, features_unseen1, features_unseen2, features_unseen3])
                    # cat_preds = torch.cat([preds_seen, preds_unseen1, preds_unseen2, preds_unseen3])
                    # labels = torch.cat([labels_seen, labels_unseen1, labels_unseen2, labels_unseen3])
                elif seen_class_group == 'HC_MDD_ABIDE':
                    # 2 BD
                    cat_features = torch.cat([features_seen, features_unseen2])
                    cat_preds = torch.cat([preds_seen, preds_unseen2])
                    labels = torch.cat([labels_seen, labels_unseen2])

                    # 3 SCHZ
                    # cat_features = torch.cat([features_seen, features_unseen3])
                    # cat_preds = torch.cat([preds_seen, preds_unseen3])
                    # labels = torch.cat([labels_seen, labels_unseen3])

                    # 23 BD_SCHZ
                    # cat_features = torch.cat([features_seen, features_unseen2, features_unseen3])
                    # cat_preds = torch.cat([preds_seen, preds_unseen2, preds_unseen3])
                    # labels = torch.cat([labels_seen, labels_unseen2, labels_unseen3])


            preds = get_outputs(cat_preds, cat_features, centroids, delta)

            # 要写入的列表数据
            data_list = preds[-76:].tolist()
            # XLSX文件路径
            # xlsx_file = r'D:\Dataset\Open\ningbo76_output.csv'     #补充实验
            # 将列表数据写入文件
            # append_list_to_csv(xlsx_file, data_list)
            # append_list_to_csv_column(xlsx_file, data_list)     #补充实验

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            print(labels)
            print(preds)
            acc_open, precision_open, specificity_open, sensitivity_open, f1_open, \
                weighted_precision, weighted_avg_sensitivity, weighted_avg_specificity, weighted_f1 = evaluate(preds, labels)

            Acc_list.append(acc_open)
            Prec_list.append(precision_open)
            Spe_list.append(specificity_open)
            Sen_list.append(sensitivity_open)
            F1_list.append(f1_open)
            W_Prec_list.append(weighted_precision)
            W_Spe_list.append(weighted_avg_specificity)
            W_Sen_list.append(weighted_avg_sensitivity)
            W_F1_list.append(weighted_f1)
            print("*********", acc_open, "****************")

    print("========== The test results ==========")
    print('\033[4;30mTotal Accuracy: %.5f %%\033[0m' % np.mean(np.array(Acc_list)))
    print('\033[4;36mTotal F1: %.5f\033[0m' % np.mean(np.array(F1_list)))
    # print('acc:', np.mean(np.array(Acc_list)), '(', np.std(np.array(Acc_list)), ')')
    # print('f1:', np.mean(np.array(F1_list)), '(', np.std(np.array(F1_list)), ')')
    print('acc: {:.4f} ({:.4f})'.format(np.mean(np.array(Acc_list)), np.std(np.array(Acc_list))))
    print('precision: {:.4f} ({:.4f})'.format(np.mean(np.array(Prec_list)), np.std(np.array(Prec_list))))
    print('weighted_precision: {:.4f} ({:.4f})'.format(np.mean(np.array(W_Prec_list)), np.std(np.array(W_Prec_list))))
    print('specificity: {:.4f} ({:.4f})'.format(np.mean(np.array(Spe_list)), np.std(np.array(Spe_list))))
    print('weighted_specificity: {:.4f} ({:.4f})'.format(np.mean(np.array(W_Spe_list)), np.std(np.array(W_Spe_list))))
    print('sensitivity: {:.4f} ({:.4f})'.format(np.mean(np.array(Sen_list)), np.std(np.array(Sen_list))))
    print('weighted_sensitivity: {:.4f} ({:.4f})'.format(np.mean(np.array(W_Sen_list)), np.std(np.array(W_Sen_list))))
    print('f1: {:.4f} ({:.4f})'.format(np.mean(np.array(F1_list)), np.std(np.array(F1_list))))
    print('weighted_f1: {:.4f} ({:.4f})'.format(np.mean(np.array(W_F1_list)), np.std(np.array(W_F1_list))))
    print(np.array(Acc_list))
    print(np.array(Prec_list))
    print(np.array(W_Prec_list))
    print(np.array(Spe_list))
    print(np.array(W_Spe_list))
    print(np.array(Sen_list))
    print(np.array(W_Sen_list))
    print(np.array(F1_list))
    print(np.array(W_F1_list))
    print(np.mean(np.array(Acc_list)))
    print(np.mean(np.array(Prec_list)))
    print(np.mean(np.array(W_Prec_list)))
    print(np.mean(np.array(Spe_list)))
    print(np.mean(np.array(W_Spe_list)))
    print(np.mean(np.array(Sen_list)))
    print(np.mean(np.array(W_Sen_list)))
    print(np.mean(np.array(F1_list)))
    print(np.mean(np.array(W_F1_list)))
