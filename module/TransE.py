import codecs
import numpy as np
import copy
import time
import random
import matplotlib.pyplot as plt
import json
import operator # operator模块输出一系列对应Python内部操作符的函数

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class TransE(nn.Module):
    def __init__(self, entity_num, relation_num, dim, margin, norm, C):
        super(TransE, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.dim = dim
        self.margin = margin
        self.norm = norm
        self.C = C

        self.ent_embedding = torch.nn.Embedding(num_embeddings=self.entity_num,
                                                          embedding_dim=self.dim).cuda()
        self.rel_embedding = torch.nn.Embedding(num_embeddings=self.relation_num,
                                                           embedding_dim=self.dim).cuda()

        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean").cuda()

        self.__data_init()

    def __data_init(self):
        # embedding.weight (Tensor) -形状为(num_embeddings, embedding_dim)的嵌入中可学习的权值
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        self.normalization_rel_embedding()
        self.normalization_ent_embedding()


    def normalization_ent_embedding(self):
        norm = self.ent_embedding.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.ent_embedding.weight.data.copy_(torch.from_numpy(norm))

    def normalization_rel_embedding(self):
        norm = self.rel_embedding.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.rel_embedding.weight.data.copy_(torch.from_numpy(norm))

    def input_pre_transe(self, ent_vector, rel_vector):
        for i in range(self.entity_num):
            self.ent_embedding.weight.data[i] = torch.from_numpy(np.array(ent_vector[i]))
        for i in range(self.relation_num):
            self.rel_embedding.weight.data[i] = torch.from_numpy(np.array(rel_vector[i]))

    def distance(self, h, r, t):
        # 在 tensor 的指定维度操作就是对指定维度包含的元素进行操作，如果想要保持结果的维度不变，设置参数keepdim=True即可
        # 如 下面sum中 r_norm * h 结果是一个1024 *50的矩阵（2维张量） sum在dim的结果就变成了 1024的向量（1位张量） 如果想和r_norm对应元素两两相乘
        # 就需要sum的结果也是2维张量 因此需要使用keepdim= True报纸张量的维度不变
        # 另外关于 dim 等于几表示张量的第几个维度，从0开始计数，可以理解为张量的最开始的第几个左括号，具体可以参考这个https://www.cnblogs.com/flix/p/11262606.html
        head = self.ent_embedding(h)
        rel = self.rel_embedding(r)
        tail = self.ent_embedding(t)

        distance = head + rel - tail
        # dim = -1表示的是维度的最后一维 比如如果一个张量有3维 那么 dim = 2 = -1， dim = 0 = -3

        score = torch.norm(distance, p = self.norm, dim=1)
        return score

    def test_distance(self, h, r, t):

        head = self.ent_embedding(h.cuda())
        rel = self.rel_embedding(r.cuda())
        tail = self.ent_embedding(t.cuda())


        distance = head + rel - tail
        # dim = -1表示的是维度的最后一维 比如如果一个张量有3维 那么 dim = 2 = -1， dim = 0 = -3

        score = torch.norm(distance, p=self.norm, dim=1)
        score = torch.autograd.Variable(torch.FloatTensor([1.0]).cuda()) / (torch.autograd.Variable(torch.FloatTensor([1.0]).cuda()) + score)
        return score.cpu().detach().numpy()

    def LP_test_distance(self, h, r, t):

        head = self.ent_embedding(h.cuda())
        rel = self.rel_embedding(r.cuda())
        tail = self.ent_embedding(t.cuda())


        distance = head + rel - tail
        # dim = -1表示的是维度的最后一维 比如如果一个张量有3维 那么 dim = 2 = -1， dim = 0 = -3

        score = torch.norm(distance, p=self.norm, dim=1)
        return score.cpu().detach().numpy()

    def scale_loss(self, embedding):
        return torch.sum(
            torch.max(
                torch.sum(
                    embedding ** 2, dim=1, keepdim=True
                )-torch.autograd.Variable(torch.FloatTensor([1.0]).cuda()),
                torch.autograd.Variable(torch.FloatTensor([0.0]).cuda())
            ))

    def forward(self, current_triples, corrupted_triples):
        h, r, t = torch.chunk(current_triples, 3, dim=1)
        h_c, r_c, t_c = torch.chunk(corrupted_triples, 3, dim=1)

        h = torch.squeeze(h, dim=1).cuda()
        r = torch.squeeze(r, dim=1).cuda()
        t = torch.squeeze(t, dim=1).cuda()
        h_c = torch.squeeze(h_c, dim=1).cuda()
        r_c = torch.squeeze(r_c, dim=1).cuda()
        t_c = torch.squeeze(t_c, dim=1).cuda()

        # torch.nn.embedding类的forward只接受longTensor类型的张量

        pos = self.distance(h, r, t)
        neg = self.distance(h_c, r_c, t_c)

        entity_embedding = self.ent_embedding(torch.cat([h, t, h_c, t_c]).cuda())
        relation_embedding = self.rel_embedding(torch.cat([r, r_c]).cuda())

        # loss_F = max(0, -y*(x1-x2) + margin)
        # loss1 = torch.sum(torch.relu(pos - neg + self.margin))
        y = Variable(torch.Tensor([-1])).cuda()
        loss = self.loss_F(pos, neg, y)

        ent_scale_loss = self.scale_loss(entity_embedding)
        rel_scale_loss = self.scale_loss(relation_embedding)
        return loss + self.C * (ent_scale_loss/len(entity_embedding) + rel_scale_loss/len(relation_embedding))