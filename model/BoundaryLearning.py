import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.neighbors import NearestNeighbors


class BoundaryLearning(nn.Module):
    def __init__(self, num_labels=2, device=None):
        super(BoundaryLearning, self).__init__()
        self.num_labels = num_labels
        self.delta = nn.Parameter(torch.randn(num_labels).to(device))
        nn.init.normal_(self.delta)
        # self.delta = nn.Parameter(torch.ones(num_labels).to(device))

    def forward(self, pooled_output, centroids, labels, density_same_class_all, density_different_class_all, epoch=None):
        x = pooled_output

        # density_same_class_all, density_different_class_all = self.get_density(x.cpu().numpy(), labels.cpu().numpy())
        # 使用密度来调整掩码
        density_factor_pos = torch.tensor(density_same_class_all / (density_same_class_all + density_different_class_all + 1e-6)).type(torch.cuda.FloatTensor)
        density_factor_neg = torch.tensor(density_different_class_all / (density_same_class_all + density_different_class_all + 1e-6)).type(torch.cuda.FloatTensor)

        delta = F.softplus(self.delta)
        c = centroids[labels]
        d = delta[labels]

        euc_dis = torch.norm(x - c, 2, 1).view(-1)
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor) * density_factor_pos
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor) * density_factor_pos
        # pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        # neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)

        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        boundary_loss = pos_loss.mean() + neg_loss.mean()
        # boundary_loss = (torch.sum(pos_loss) / torch.sum(pos_mask) + torch.sum(neg_loss) / torch.sum(pos_mask)) * self.reg_factor
        # boundary_loss = torch.sum(pos_loss) / torch.sum(pos_mask) + torch.sum(neg_loss) / torch.sum(pos_mask)

        # 添加动态正则化项
        regularization_weight = self.dynamic_regularization_weight()
        regularization_loss = regularization_weight * torch.mean(delta)

        loss = boundary_loss + regularization_loss
        if epoch == 700:
            print('debug')

        return loss, delta

    def dynamic_regularization_weight(self):
        # 根据类别数量动态调整正则化权重
        if self.num_labels == 2:
            # 0.97
            # return 0.485 * self.num_labels
            # 0.96
            # return 0.48 * self.num_labels
            # 0.95
            # return 0.475 * self.num_labels
            return 0.9  # pre
            # return 0.4  # supplement 0.7
        elif self.num_labels == 3:
            return 0.4  # pre
            # return 0.2    # supplement 0.7
        elif self.num_labels == 4:
            return 0
        else:
            # 类别较多时，减少对delta的惩罚
            # 这里使用一个简单的公式，实际应用中可以根据需要调整
            return 0.5 / (self.num_labels - 1)

    # def calculate_reg_factor(self, num_labels):
    #     # 这里使用对数函数作为示例，实际可以根据需要调整
    #     return torch.log(torch.tensor(num_labels, dtype=torch.float)).to('cuda')
    #
    # def calculate_reg_factor(self, num_labels):
    #     # 示例：使用类别数量的倒数作为调节因子
    #     return 1.0 / num_labels

    def get_density(self, X, y, k=5):
        # 初始化 NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1)  # +1因为包括样本点本身
        nn.fit(X)

        # 存储每个样本的密度
        density_same_class_all = np.zeros(len(X))
        density_different_class_all = np.zeros(len(X))

        # 遍历每个样本计算密度
        for sample_index in range(len(X)):
            sample_point = X[sample_index].reshape(1, -1)
            sample_label = y[sample_index]

            # 计算最近邻
            distances, indices = nn.kneighbors(sample_point)

            # 使用距离的倒数作为权重计算密度
            weights = 1 / np.maximum(distances, 1e-10)

            # 筛选出同类和其他类的权重
            same_class_weights = weights[0][1:][(y[indices[0][1:]] == sample_label)]
            different_class_weights = weights[0][1:][(y[indices[0][1:]] != sample_label)]

            # 计算加权密度
            density_same_class_all[sample_index] = np.sum(same_class_weights)
            density_different_class_all[sample_index] = np.sum(different_class_weights)

        return density_same_class_all, density_different_class_all


