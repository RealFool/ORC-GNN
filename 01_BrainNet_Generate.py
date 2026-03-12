import torch
import os
from os import listdir
import scipy.io as scio
import numpy as np
from sklearn.metrics import r2_score
from Glasso import GroupLasso
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import networkx as nx

GroupLasso.LOG_LOSSES = False
cwd = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--Openset_pre', type=bool, default=True, help='Is open set recognition data preprocessing enabled')
# parser.add_argument('--Fun_dir_open', type=str, default=os.path.join(cwd, "data/Functional/openset"), help='Functional_direction')
# parser.add_argument('--Save_dir_open', type=str, default=os.path.join(cwd, "data/BrainNet/openset"), help='Network_direction')
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet"), help='Network_direction')

# 补充实验
parser.add_argument('--Fun_dir_open', type=str, default=os.path.join(cwd, "data/supplement2/Functional/openset"), help='Functional_direction')
parser.add_argument('--Save_dir_open', type=str, default=os.path.join(cwd, "data/supplement2/BrainNet/openset"), help='Network_direction')
# 补充实验，HC,MDD,ABIDE多样本，已知类HC_MDD
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/supplement3/Functional/closeset_HC_MDD"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/supplement3/BrainNet_HC_MDD"), help='Network_direction')
# 补充实验，HC,MDD,ABIDE多样本，已知类HC_ABIDE
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/supplement3/Functional/closeset_HC_ABIDE"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/supplement3/BrainNet_HC_ABIDE"), help='Network_direction')
parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/supplement2/Functional/closeset_HC_MDD_ABIDE"), help='Functional_direction')
parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/supplement2/BrainNet_HC_MDD_ABIDE"), help='Network_direction')


# 已知类：HC_MDD_BD_SCHZ
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet"), help='Network_direction')
# 已知类：HC\BD\SCHZ
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset_HC_BD_SCHZ"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet_HC_BD_SCHZ"), help='Network_direction')
# 已知类：HC\MDD\BD
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset_HC_MDD_BD"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet_HC_MDD_BD"), help='Network_direction')
# 已知类：HC\MDD
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset_HC_MDD"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet_HC_MDD"), help='Network_direction')
# 已知类：HC\BD
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset_HC_BD"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet_HC_BD"), help='Network_direction')

# 已知类：HC_MDD_BD_ABIDE
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset_HC_MDD_BD_ABIDE"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet_HC_MDD_BD_ABIDE"), help='Network_direction')
# 已知类：HC_MDD_SCHZ_ABIDE
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset_HC_MDD_SCHZ_ABIDE"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet_HC_MDD_SCHZ_ABIDE"), help='Network_direction')

# 已知类：HC_SCHZ_ABIDE
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset_HC_SCHZ_ABIDE"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet_HC_SCHZ_ABIDE"), help='Network_direction')
# 已知类：HC_BD_ABIDE
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset_HC_BD_ABIDE"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet_HC_BD_ABIDE"), help='Network_direction')

# 已知类：HC_SCHZ
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset_HC_SCHZ"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet_HC_SCHZ"), help='Network_direction')
# 已知类：HC_ABIDE
# parser.add_argument('--Fun_dir', type=str, default=os.path.join(cwd, "data/Functional/closeset_HC_ABIDE"), help='Functional_direction')
# parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet_HC_ABIDE"), help='Network_direction')

parser.add_argument('--ROI_risk', type=str, default=os.path.join(cwd, "data/Risk_ROI_List.csv"), help='Risk ROI')
# parser.add_argument('--Lambda_group', type=float, default=0.05, help='Lasso Constraint')
parser.add_argument('--Lambda_group', type=str, default='0.3_0.5', help='Lasso Constraint')
parser.add_argument('--Thres', type=float, default=0.01, help='Reserve Reliable Connections')
parser.add_argument('--Vis_Loss', type=bool, default=False, help='Visualize the Regularization')
parser.add_argument('--BiGraph_Ratio', type=float, default=0.5, help='Build a bipartite graph by ratio')  # 0.1
parser.add_argument('--WoGraph_Ratio', type=float, default=0.3, help='Reserve Reliable Connections')
opt = parser.parse_args()

# 补充实验
if opt.Openset_pre:
    class_name = ['HC', 'MDD', 'ABIDE', 'ningbo', 'ningbo76']
    # unseen 25%
    unseen_class = class_name[4]
    opt.Fun_dir = os.path.join(opt.Fun_dir_open, unseen_class)
    opt.Save_dir = os.path.join(opt.Save_dir_open, unseen_class)
# if opt.Openset_pre:
#     class_name = ['HC', 'MDD', 'BD', 'SCHZ', 'ABIDE']
#     # unseen 25%
#     unseen_class = class_name[4]
#     opt.Fun_dir = os.path.join(opt.Fun_dir_open, unseen_class)
#     opt.Save_dir = os.path.join(opt.Save_dir_open, unseen_class)
Funfiles = [f for f in listdir(opt.Fun_dir) if os.path.isfile(os.path.join(opt.Fun_dir, f))]
Funfiles.sort()

def compute_grouplasso(A, y, ROItoreg, lam_group, lam_l1=0, Vis_Loss=opt.Vis_Loss):
    ROI_risk = opt.ROI_risk
    if not os.path.isfile(ROI_risk):
        print(ROI_risk + 'is not found!')
        print('ROI grouping information does not exist!')
        ROI_group = np.zeros(116, 1)
    else:
        if ROI_risk.endswith('.csv'):
            ROI_group = np.genfromtxt(ROI_risk, dtype=int,
                                      delimiter=',', skip_header=1, usecols=(3)).reshape(-1, 1)
    ROI_g = np.delete(ROI_group, ROItoreg, axis=0)
    gl = GroupLasso(groups=ROI_g, group_reg=lam_group, l1_reg=lam_l1,
                    frobenius_lipschitz=True,
                    scale_reg="group_size", subsampling_scheme=1,
                    supress_warning=True, n_iter=1000, tol=1e-3)  # scale_reg="inverse_group_size
    gl.fit(A, y)
    yhat = gl.predict(A)
    sparsity_mask = gl.sparsity_mask_
    w_hat = gl.coef_
    R2 = r2_score(y, yhat)
    if Vis_Loss:
        print("Group lasso computing...")
        print(f"Number variables: {len(sparsity_mask)}")
        print(f"Number of chosen variables: {sparsity_mask.sum()}")
        print(f"performance metrics R^2: {R2}")
        plt.figure()
        plt.plot(gl.losses_);
        plt.title("Loss plot")
        plt.ylabel("Mean squared error");
        plt.xlabel("Iteration")
        plt.show()

    return w_hat


def get_wholebrain_networks(subjects_fun, lam_group=0.1, binary_require=False, variable_fun='ROISignals', save=True,
                            save_path=opt.Save_dir):
    for ROI_series in subjects_fun:
        fun_position = os.path.join(opt.Fun_dir, ROI_series)
        timeseries = scio.loadmat(fun_position)[variable_fun]
        # if timeseries.shape[1] == 116:
        #     # 去除小脑区域
        #     timeseries = timeseries[:, 0:90]

        w_matrix = np.corrcoef(np.transpose(timeseries))
        # 将邻接矩阵 A 的对角线元素置为0
        w_matrix = w_matrix - np.multiply(w_matrix, np.eye(w_matrix.shape[0]))

        # NumROI = timeseries.shape[1]
        # w_matrix_list = []
        # for j in range(NumROI):
        #     y = timeseries[:, j]
        #     A = np.delete(timeseries, j, axis=1)
        #     w_coeff = compute_grouplasso(A, y, j, lam_group)
        #     w_hat_res = np.insert(w_coeff, j, [0])
        #     w_matrix_list.append(w_hat_res)
        # w_matrix = np.array(w_matrix_list, dtype=float)

        if binary_require != True:
            Brain_temp = (w_matrix + w_matrix.T) / 2
            # Brain_spa_mat = abs(Brain_temp)

            # 根据给定阈值过滤矩阵
            Brain_spa_mat = np.where(abs(Brain_temp) > opt.Thres, Brain_temp, 0)

            # 应用动态阈值过滤矩阵
            # all_values = np.abs(Brain_temp).flatten()
            # threshold_value = np.percentile(all_values, (1 - opt.WoGraph_Ratio) * 100)
            # Brain_spa_mat = np.where(np.abs(Brain_temp) >= threshold_value, Brain_temp, 0)
        else:
            Brain_spa_mat = np.int64((w_matrix + w_matrix.T) / 2 != 0)

        if save:
            Folder_path = os.path.join(save_path, str(lam_group), 'raw')
            # Folder_path = os.path.join(save_path, str(lam_group), 'raw2')
            if not os.path.exists(Folder_path):
                os.makedirs(Folder_path)
            scio.savemat(os.path.join(Folder_path, '%s_net_%s.mat' % (ROI_series[:-4], str(lam_group))),
                         {'Brainnetwork': abs(Brain_spa_mat)})
            print("===========================")
            print("Save Brain_network: %s_net_%s" % (ROI_series[:-4], str(lam_group)))

    print("===========================")
    print("Multimodal Brain network completed")


def get_harfbrain_networks(subjects_fun, lam_group=0.1, binary_require=False, variable_fun='ROISignals', save=True,
                           save_path=opt.Save_dir):
    for ROI_series in subjects_fun:
        fun_position = os.path.join(opt.Fun_dir, ROI_series)
        timeseries = scio.loadmat(fun_position)[variable_fun]
        whole_fMRI_cor = np.corrcoef(np.transpose(timeseries))
        if timeseries.shape[1] == 116:
            # 去除小脑区域
            timeseries = timeseries[:, 0:108]
        # 拆分数组为奇数列（左脑）和偶数列（右脑）
        left_fMRI_timeseries = timeseries[:, 0::2]  # 奇数列
        right_fMRI_timeseries = timeseries[:, 1::2]  # 偶数列

        # 构造二分图
        edges = build_bipartite_graph(np.transpose(left_fMRI_timeseries), np.transpose(right_fMRI_timeseries))
        # 构造二分图邻接矩阵
        bipartite_adjacency_matrix = build_bipartite_adjacency_matrix(left_fMRI_timeseries.shape[1] * 2, edges, extra_padding=8)
        bipartite_threshold_value = np.percentile(np.abs(bipartite_adjacency_matrix).flatten(), (1 - opt.BiGraph_Ratio) * 100)
        bipartite_spa_mat = np.where(np.abs(bipartite_adjacency_matrix) >= bipartite_threshold_value, bipartite_adjacency_matrix, 0)

        left_fMRI_cor = np.corrcoef(np.transpose(left_fMRI_timeseries))
        right_fMRI_cor = np.corrcoef(np.transpose(right_fMRI_timeseries))
        # 还原shape
        left_fMRI_cor_reshape = reshape_matrix(left_fMRI_cor, zeros_on_odd=False, extra_padding=8)
        right_fMRI_cor_reshape = reshape_matrix(right_fMRI_cor, zeros_on_odd=True, extra_padding=8)
        # 将对角线元素置为0
        left_fMRI_w_matrix = left_fMRI_cor_reshape - np.multiply(left_fMRI_cor_reshape,
                                                                 np.eye(left_fMRI_cor_reshape.shape[0]))
        right_fMRI_w_matrix = right_fMRI_cor_reshape - np.multiply(right_fMRI_cor_reshape,
                                                                   np.eye(right_fMRI_cor_reshape.shape[0]))

        if binary_require != True:
            Left_Brain_temp = (left_fMRI_w_matrix + left_fMRI_w_matrix.T) / 2
            # Left_Brain_spa_mat = abs(Left_Brain_temp)
            # Left_Brain_spa_mat = np.where(abs(Left_Brain_temp) > opt.Thres, Left_Brain_temp, 0)
            left_threshold_value = np.percentile(np.abs(Left_Brain_temp).flatten(), (1 - opt.BiGraph_Ratio) * 100)
            Left_Brain_spa_mat = np.where(np.abs(Left_Brain_temp) >= left_threshold_value, Left_Brain_temp, 0)

            Right_Brain_temp = (right_fMRI_w_matrix + right_fMRI_w_matrix.T) / 2
            # Right_Brain_spa_mat = abs(Right_Brain_temp)
            # Right_Brain_spa_mat = np.where(abs(Right_Brain_temp) > opt.Thres, Right_Brain_temp, 0)
            right_threshold_value = np.percentile(np.abs(Right_Brain_temp).flatten(), (1 - opt.BiGraph_Ratio) * 100)
            Right_Brain_spa_mat = np.where(np.abs(Right_Brain_temp) >= right_threshold_value, Right_Brain_temp, 0)
        else:
            Left_Brain_spa_mat = np.int64((left_fMRI_w_matrix + left_fMRI_w_matrix.T) / 2 != 0)
            Right_Brain_spa_mat = np.int64((left_fMRI_w_matrix + left_fMRI_w_matrix.T) / 2 != 0)

        if save:
            Folder_path = os.path.join(save_path, str(lam_group), 'raw', 'harfbrain', 'raw')
            if not os.path.exists(Folder_path):
                os.makedirs(Folder_path)
            scio.savemat(os.path.join(Folder_path, '%s_net_%s.mat' % (ROI_series[:-4], 'harfbrain_network')),
                         {'LeftBrainNetwork': abs(Left_Brain_spa_mat), 'RightBrainNetwork': abs(Right_Brain_spa_mat),
                          'LeftBrainAtt': abs(left_fMRI_cor_reshape), 'RightBrainAtt': abs(right_fMRI_cor_reshape),
                          'BipartiteBrainNetwork': bipartite_spa_mat, 'BipartiteBrainAtt': whole_fMRI_cor})

            print("===========================")
            print("Save Harf_Brain_network: %s_net_%s" % (ROI_series[:-4], 'harfbrain_network'))


def reshape_matrix(A, zeros_on_odd=False, extra_padding=0):
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


def calculate_mutual_information_continuous(series1, series2):
    """
    Calculate the mutual information between two continuous time series.
    """
    # Reshape the data as mutual_info_regression expects 2D arrays
    series1 = series1.reshape(-1, 1)
    series2 = series2.reshape(-1, 1)

    # Calculate mutual information
    mi = mutual_info_regression(series1, series2.ravel())
    return mi[0]  # Return the mutual information value


def build_bipartite_graph(groupA, groupB, ratio=1):
    """
    Build a bipartite graph based on mutual information for continuous data,
    by jointly sorting the mutual information of all possible pairs from groupA and groupB.
    Connect a certain ratio of the top pairs.
    """
    edges = []
    all_pairs = []

    # Calculate mutual information for all pairs
    for i, seriesA in enumerate(groupA):
        for j, seriesB in enumerate(groupB):
            mi = calculate_mutual_information_continuous(seriesA, seriesB)
            all_pairs.append(((i, j), mi))

    # Sort all pairs by mutual information
    sorted_pairs = sorted(all_pairs, key=lambda x: x[1], reverse=True)

    # Calculate the number of edges to create based on the ratio
    num_edges = int(len(all_pairs) * ratio)

    # Get the top pairs based on the ratio
    top_pairs = sorted_pairs[:num_edges]

    # Create edges
    edges = [((i, j), mi_value) for (i, j), mi_value in top_pairs]

    # 绘制
    # plot_bipartite_graph(edges, 45, 45)

    return edges


def build_bipartite_adjacency_matrix(total_size, edges, extra_padding=0):
    """
    Build a bipartite adjacency matrix for groups A and B,
    where each off-diagonal block represents the mutual information between nodes of different groups.
    """
    # Calculate the size of the new matrix
    new_size = total_size + extra_padding
    adjacency_matrix = np.zeros((new_size, new_size))

    # Fill the adjacency matrix with mutual information values for edges between group A and group B
    for (i, j), mi_value in edges:
        # 左右脑结点定位，还原
        i_adj = fill_position(i, fill_odd=True)
        j_adj = fill_position(j, fill_odd=False)
        # j_adj = j + int(total_size / 2)
        adjacency_matrix[i_adj, j_adj] = mi_value
        adjacency_matrix[j_adj, i_adj] = mi_value  # Ensure the matrix is symmetric

    return adjacency_matrix


def fill_position(i, fill_odd=True):
    if fill_odd:
        # 到奇数位置 (数组索引为偶数的位置)
        position = i * 2
    else:
        # 到偶数位置 (数组索引为奇数的位置)
        position = i * 2 + 1
    return position


def plot_bipartite_graph(edges, groupA_size, groupB_size):
    """
    Plot the bipartite graph using the edges list, with manual position assignment.
    """
    # Create a new graph
    B = nx.Graph()

    # Add nodes with group labels
    B.add_nodes_from(range(groupA_size), bipartite=0)  # Group A
    B.add_nodes_from(range(groupA_size, groupA_size + groupB_size), bipartite=1)  # Group B

    # Add edges
    B.add_edges_from([(a, b + groupA_size) for a, b in edges])  # Adjust groupB node indices

    # Position the nodes in two columns
    pos = {}
    pos.update((n, (1, i)) for i, n in enumerate(range(groupA_size)))  # Group A
    pos.update((n, (2, i)) for i, n in enumerate(range(groupA_size, groupA_size + groupB_size)))  # Group B

    # Draw the graph
    plt.figure(figsize=(24, 16))
    nx.draw(B, pos, with_labels=True, node_color=['lightblue' if n < groupA_size else 'lightgreen' for n in B.nodes()])
    plt.show()


def main():
    get_wholebrain_networks(Funfiles, opt.Lambda_group)
    get_harfbrain_networks(Funfiles, opt.Lambda_group)


if __name__ == '__main__':
    main()
