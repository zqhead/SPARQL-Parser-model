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

class TransR(nn.Module):
    def __init__(self, entity_num, relation_num, ent_dim, rel_dim, margin, norm, C):
        super(TransR, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.margin = margin
        self.norm = norm
        self.C = C


        self.ent_embedding = torch.nn.Embedding(num_embeddings=self.entity_num,
                                                          embedding_dim=self.ent_dim).cuda()
        self.rel_embedding = torch.nn.Embedding(num_embeddings=self.relation_num,
                                                           embedding_dim=self.rel_dim).cuda()
        self.rel_matrix = torch.nn.Embedding(num_embeddings= self.relation_num,
                                                           embedding_dim=self.ent_dim*self.rel_dim).cuda()
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean").cuda()

        self.__data_init()

    def __data_init(self):
        # embedding.weight (Tensor)
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        identity = torch.zeros(self.ent_dim, self.rel_dim)
        for i in range(min(self.ent_dim, self.rel_dim)):
            identity[i][i] = 1
        identity = identity.view(self.ent_dim * self.rel_dim)
        for i in range(self.relation_num):
            self.rel_matrix.weight.data[i] = identity

    def input_pre_transe(self, ent_vector, rel_vector):
        for i in range(self.entity_num):
            self.ent_embedding.weight.data[i] = torch.from_numpy(np.array(ent_vector[i]))
        for i in range(self.relation_num):
            self.rel_embedding.weight.data[i] = torch.from_numpy(np.array(rel_vector[i]))

    def input_pre_transr(self, ent_vector, rel_vector, rel_matrix):
        for i in range(self.entity_num):
            self.ent_embedding.weight.data[i] = torch.from_numpy(np.array(ent_vector[i]))
        for i in range(self.relation_num):
            self.rel_embedding.weight.data[i] = torch.from_numpy(np.array(rel_vector[i]))
        for i in range(self.relation_num):
            self.rel_matrix.weight.data[i] = torch.from_numpy(np.array(rel_matrix[i]))

    def transfer(self, e, rel_mat):
        rel_matrix = rel_mat.view(-1, self.ent_dim, self.rel_dim)
        e = e.view(-1, 1, self.ent_dim)
        e = torch.matmul(e, rel_matrix)

        return e.view(-1, self.rel_dim)

    def distance(self, h, r, t):

        head = self.ent_embedding(h)
        rel = self.rel_embedding(r)
        rel_mat = self.rel_matrix(r)
        tail = self.ent_embedding(t)

        head = self.transfer(head, rel_mat)
        tail = self.transfer(tail, rel_mat)

        head = F.normalize(head, 2, -1)
        rel = F.normalize(rel, 2, -1)
        tail = F.normalize(tail, 2, -1)
        distance = head + rel - tail


        score = torch.norm(distance, p = self.norm, dim=1)
        return score

    def test_distance(self, h, r, t):

        head = self.ent_embedding(h.cuda())
        rel = self.rel_embedding(r.cuda())
        rel_mat = self.rel_matrix(r.cuda())
        tail = self.ent_embedding(t.cuda())

        head = self.transfer(head, rel_mat)
        tail = self.transfer(tail, rel_mat)


        distance = head + rel - tail


        score = torch.norm(distance, p=self.norm, dim=1)
        score = torch.autograd.Variable(torch.FloatTensor([1.0]).cuda()) / (torch.autograd.Variable(torch.FloatTensor([1.0]).cuda()) + score)
        return score.cpu().detach().numpy()

    def LP_test_distance(self, h, r, t):

        head = self.ent_embedding(h.cuda())
        rel = self.rel_embedding(r.cuda())
        rel_mat = self.rel_matrix(r.cuda())
        tail = self.ent_embedding(t.cuda())

        head = self.transfer(head, rel_mat)
        tail = self.transfer(tail, rel_mat)


        distance = head + rel - tail

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

        entity_embedding = self.ent_embedding(torch.cat([h, t, h_c, t_c]).cuda())
        relation_embedding = self.rel_embedding(torch.cat([r, r_c]).cuda())



        pos = self.distance(h, r, t)
        neg = self.distance(h_c, r_c, t_c)

        # loss_F = max(0, -y*(x1-x2) + margin)
        y = Variable(torch.Tensor([-1])).cuda()
        loss = self.loss_F(pos, neg, y)

        ent_scale_loss = self.scale_loss(entity_embedding) / len(entity_embedding)
        rel_scale_loss = self.scale_loss(relation_embedding) / len(relation_embedding)
        return loss + self.C * (ent_scale_loss + rel_scale_loss)