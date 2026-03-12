import os
from utils.brainnetwork_reader import MyNetworkReader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import deepdish as dd
import torch
import argparse
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PH layer')
parser.add_argument('--Openset_pre', type=bool, default=True, help='Is open set recognition data preprocessing enabled')

parser.add_argument('--nROI', type=int, default=116, help='Number of ROIs in AAL template')
parser.add_argument('--resolution', type=int, default=1000, help='Resolution in persistence landscape')
parser.add_argument('--land', type=int, default=8, help='layers in PL')
parser.add_argument('--dim_obj', type=int, default=0, help='methods in PH layer')
# parser.add_argument('--lam_group', type=float, default=0.05, help='regularization parameters')
parser.add_argument('--lam_group', type=str, default='0.3_0.5', help='regularization parameters')
# parser.add_argument('--batch', type=int, default=196, help='Batch_size')
parser.add_argument('--batch', type=int, default=1500, help='Batch_size')
args = parser.parse_args()

nROI = args.nROI
resolution = args.resolution
land = args.land
dim_obj = args.dim_obj
lam_group = args.lam_group
batch = args.batch

# 补充实验 supplement
if args.Openset_pre:
    class_name = ['HC', 'MDD', 'ABIDE', 'ningbo', 'ningbo76']   # 补充实验
    unseen_class = class_name[4]
    BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "supplement2", "BrainNet", "openset", unseen_class), str(lam_group))
    data_dir = os.path.join(os.getcwd(), "data", "supplement2", "UpperFeat", "openset", unseen_class)
# if args.Openset_pre:
#     class_name = ['HC', 'MDD', 'BD', 'SCHZ', 'ABIDE']
#     # unseen 25%
#     unseen_class = class_name[1]
#     BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet", "openset", unseen_class), str(lam_group))
#     data_dir = os.path.join(os.getcwd(), "data", "UpperFeat", "openset", unseen_class)
else:
    # 补充实验，HC,MDD,ABIDE多样本，已知类HC_MDD
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "supplement3", "BrainNet_HC_MDD"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "supplement3", "UpperFeat_HC_MDD")
    # 补充实验，HC,MDD,ABIDE多样本，已知类HC_ABIDE
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "supplement3", "BrainNet_HC_ABIDE"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "supplement3", "UpperFeat_HC_ABIDE")
    # 补充实验，HC,MDD,ABIDE多样本，已知类HC_MDD_ABIDE
    BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "supplement3", "BrainNet_HC_MDD_ABIDE"), str(lam_group))
    data_dir = os.path.join(os.getcwd(), "data", "supplement3", "UpperFeat_HC_MDD_ABIDE")

    # 默认HC\MDD\BD\SCHZ
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "UpperFeat")
    # 已知类：HC\BD\SCHZ
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet_HC_BD_SCHZ"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "UpperFeat_HC_BD_SCHZ")
    # 已知类：HC\MDD\BD
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet_HC_MDD_BD"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "UpperFeat_HC_MDD_BD")
    # 已知类：HC\MDD
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet_HC_MDD"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "UpperFeat_HC_MDD")
    # 已知类：HC\BD
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet_HC_BD"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "UpperFeat_HC_BD")

    # 已知类：HC_MDD_BD_ABIDE
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet_HC_MDD_BD_ABIDE"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "UpperFeat_HC_MDD_BD_ABIDE")
    # 已知类：HC_MDD_SCHZ_ABIDE
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet_HC_MDD_SCHZ_ABIDE"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "UpperFeat_HC_MDD_SCHZ_ABIDE")
    # 已知类：HC_SCHZ_ABIDE
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet_HC_SCHZ_ABIDE"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "UpperFeat_HC_SCHZ_ABIDE")
    # 已知类：HC_BD_ABIDE
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet_HC_BD_ABIDE"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "UpperFeat_HC_BD_ABIDE")
    # 已知类：HC_SCHZ
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet_HC_SCHZ"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "UpperFeat_HC_SCHZ")
    # 已知类：HC_ABIDE
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet_HC_ABIDE"), str(lam_group))
    # data_dir = os.path.join(os.getcwd(), "data", "UpperFeat_HC_ABIDE")

if not os.path.exists(os.path.join(data_dir)):
    os.makedirs(os.path.join(data_dir))


def main(brain_ratio_str):
    # 构边比例消融
    # BrainNetwork_dir = os.path.join(os.path.join(os.getcwd(), "data", "BrainNet"), brain_ratio_str)

    dataset = MyNetworkReader(BrainNetwork_dir)
    dataset.data.y = dataset.data.y.squeeze()
    Sub_list = dataset.subject
    data_loader = DataLoader(dataset, batch_size=batch, pin_memory=False, shuffle=False, collate_fn=custom_collate)
    seq_idx = 0
    num_iter = 0
    for data in data_loader:
        data = data.to(device)
        # Topological layer
        Subject_ID = Sub_list[seq_idx:data.num_graphs + seq_idx]
        seq_idx = seq_idx + data.num_graphs
        for i, subject in enumerate(Subject_ID):
            dd.io.save(os.path.join(data_dir, subject + '_uf' + '.h5'),
                       {'id': subject, 'UpperFeat': torch.from_numpy(np.array(dataset.upper_triangle_feature[i]))})
        num_iter = num_iter + 1
        print("Batch %s completed for saving upper features" % num_iter)


def custom_collate(data_list):
    # 为每个属性创建空列表
    x, edge_index, edge_attr, y, pos = [], [], [], [], []
    left_x, left_edge_index, left_edge_attr, left_pos = [], [], [], []
    right_x, right_edge_index, right_edge_attr, right_pos = [], [], [], []
    bipartite_x, bipartite_edge_index, bipartite_edge_attr, bipartite_pos = [], [], [], []

    for data in data_list:
        # 添加每个Data对象的属性到相应的列表
        x.append(data.x)
        edge_index.append(data.edge_index)
        edge_attr.append(data.edge_attr)
        y.append(data.y)
        pos.append(data.pos)

        left_x.append(data.left_x)
        left_edge_index.append(data.left_edge_index)
        left_edge_attr.append(data.left_edge_attr)
        left_pos.append(data.left_pos)

        right_x.append(data.right_x)
        right_edge_index.append(data.right_edge_index)
        right_edge_attr.append(data.right_edge_attr)
        right_pos.append(data.right_pos)

        bipartite_x.append(data.bipartite_x)
        bipartite_edge_index.append(data.bipartite_edge_index)
        bipartite_edge_attr.append(data.bipartite_edge_attr)
        bipartite_pos.append(data.bipartite_pos)

    # 使用Batch.from_data_list将每个属性列表合并为批次
    batch = Batch.from_data_list(data_list)
    batch.x = torch.cat(x, dim=0)
    batch.edge_index = torch.cat(edge_index, dim=1)
    batch.edge_attr = torch.cat(edge_attr, dim=0)
    batch.y = torch.cat(y, dim=0)
    batch.pos = torch.cat(pos, dim=0)

    batch.left_x = torch.cat(left_x, dim=0)
    batch.left_edge_index = torch.cat(left_edge_index, dim=1)
    batch.left_edge_attr = torch.cat(left_edge_attr, dim=0)
    batch.left_pos = torch.cat(left_pos, dim=0)

    batch.right_x = torch.cat(right_x, dim=0)
    batch.right_edge_index = torch.cat(right_edge_index, dim=1)
    batch.right_edge_attr = torch.cat(right_edge_attr, dim=0)
    batch.right_pos = torch.cat(right_pos, dim=0)

    batch.bipartite_x = torch.cat(bipartite_x, dim=0)
    batch.bipartite_edge_index = torch.cat(bipartite_edge_index, dim=1)
    batch.bipartite_edge_attr = torch.cat(bipartite_edge_attr, dim=0)
    batch.bipartite_pos = torch.cat(bipartite_pos, dim=0)

    return batch


if __name__ == '__main__':
    # wo_brain_ratio = ['0.1', '0.3', '0.5', '0.7', '0.9']
    # # bi_brain_ratio = ['0.1', '0.3', '0.5', '0.7', '0.9']
    # bi_brain_ratio = ['0.1', '0.2', '0.3', '0.4', '0.5']
    # for i in range(len(wo_brain_ratio)):
    #     for j in range(len(bi_brain_ratio)):
    #         brain_ratio_str = wo_brain_ratio[i] + '_' + bi_brain_ratio[j]
    #         main(brain_ratio_str)
    main('0.3_0.5')
