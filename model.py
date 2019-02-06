import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import os
import numpy as np
import math
import scipy.sparse as sp


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class Fc_ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=False, num_bottleneck=512):
        super(Fc_ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f


class ReFineBlock(nn.Module):
    def __init__(self, input_dim=512, dropout=True, relu=True, num_bottleneck=512, layer=2):
        super(ReFineBlock, self).__init__()
        add_block = []
        for i in range(layer):
            add_block += [nn.Linear(input_dim, num_bottleneck)]
            add_block += [nn.BatchNorm1d(num_bottleneck)]
            if relu:
                add_block += [nn.LeakyReLU(0.1)]
            if dropout:
                add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc = add_block

    def forward(self, x):
        x = self.fc(x)
        return x


class FcBlock(nn.Module):
    def __init__(self, input_dim=512, dropout=True, relu=True, num_bottleneck=512):
        super(FcBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc = add_block

    def forward(self, x):
        x = self.fc(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim=512, class_num=751):
        super(ClassBlock, self).__init__()
        classifier = []
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = Fc_ClassBlock(2048, class_num, dropout=0.5, relu=False)
        # remove the final downsample
        # self.model.layer4[0].downsample[0].stride = (1,1)
        # self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x, f = self.classifier(x)
        return x, f


# Define a 2048 to 2 Model
class verif_net(nn.Module):
    def __init__(self):
        super(verif_net, self).__init__()
        self.classifier = Fc_ClassBlock(512, 2, dropout=0.75, relu=False)

    def forward(self, x):
        x = self.classifier.classifier(x)
        return x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num=751):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = Fc_ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = Fc_ClassBlock(2048 + 1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0, x1), 1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 6  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, Fc_ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        # y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


class PCB_test(nn.Module):
    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


# debug model structure
# net = ft_net(751)
# net = ft_net(751)
# print(net)
# input = Variable(torch.FloatTensor(8, 3, 224, 224))
# output,f = net(input)
# print('net output size:')
# print(f.shape)

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.classifier = Fc_ClassBlock(512, 2, dropout=0.75, relu=False)

    def forward(self, x1, x2=None):
        output1, feature1 = self.embedding_net(x1)
        if x2 is None:
            return output1, feature1
        output2, feature2 = self.embedding_net(x2)
        feature = (feature1 - feature2).pow(2)

        # f_norm = feature.norm(p=2, dim=1, keepdim=True) + 1e-8
        # feature = feature.div(f_norm)

        result = self.classifier.classifier(feature)
        return output1, feature1, output2, feature2, feature, result


    def get_embedding(self, x):
        return self.embedding_net(x)


class Sggnn_siamese(nn.Module):
    def __init__(self, siamesemodel, hard_weight=True):
        super(Sggnn_siamese, self).__init__()
        self.basemodel = siamesemodel
        self.hard_weight = hard_weight

    def forward(self, x, y=None):
        use_gpu = torch.cuda.is_available()
        batch_size = len(x)
        x_p = x[:, 0]
        x_p = x_p.unsqueeze(1)
        x_g = x[:, 1:]
        num_img = len(x[0])
        num_p_per_id = len(x_p[0])  # 1
        num_g_per_id = len(x_g[0])  # 3
        num_p_per_batch = len(x_p) * len(x_p[0])  # 8
        num_g_per_batch = len(x_g) * len(x_g[0])  # 24
        len_feature = 512
        d = torch.FloatTensor(batch_size, batch_size, num_p_per_id, num_g_per_id, len_feature).zero_()
        # this w for dynamic calculate the weight
        # w = torch.FloatTensor(batch_size, batch_size, num_g_per_id, num_g_per_id, 1).zero_()
        # this w for calculate the weight by label
        w = torch.FloatTensor(batch_size, batch_size, num_g_per_id, num_g_per_id).zero_()
        label = torch.LongTensor(batch_size, batch_size, num_p_per_id, num_g_per_id).zero_()

        if use_gpu:
            d = d.cuda()
            w = w.cuda()
            label = label.cuda()
        if y is not None:
            y_p = y[:, 0]
            y_p = y_p.unsqueeze(1)
            y_g = y[:, 1:]

        # print('batch_size = %d  num_p_per_batch = %d  num_g_per_batch = %d' % (batch_size, num_p_per_batch, num_g_per_batch))
        for k in range(batch_size):
            x_g_temp1 = x_g[:k]
            x_g_temp2 = x_g[k:]
            x_g_temp = torch.cat((x_g_temp2, x_g_temp1), 0)
            if y is not None:
                y_temp1 = y_g[:k]
                y_temp2 = y_g[k:]
                y_temp = torch.cat((y_temp2, y_temp1), 0)

            for i in range(num_p_per_id):
                for j in range(num_g_per_id):
                    d[k, :, i, j] = self.basemodel(x_p[:, i], x_g_temp[:, j])[-2]
                    if y is not None:
                        label[k, :, i, j] = torch.where(y_p[:, i] == y_temp[:, j], torch.full_like(y_p[:, i], 1),
                                                        torch.full_like(y_p[:, i], 0))

            for i in range(num_g_per_id):
                for j in range(num_g_per_id):
                    if self.hard_weight and y is not None:
                        w[k, :, i, j] = torch.where(y_g[:, i] == y_temp[:, j], torch.full_like(y_g[:, i], 1),
                                                    torch.full_like(y_g[:, i], 0))
                    else:
                        w[k, :, i, j] = self.basemodel(x_g[:, i], x_g_temp[:, j])[-1]

        print('run Sggnn_siamese foward success  !!!')
        if y is not None:
            return d, w, label
        else:
            return d, w


class Sggnn_gcn(nn.Module):
    def __init__(self):
        super(Sggnn_gcn, self).__init__()
        self.rf = ReFineBlock(layer=2)
        # self.fc = FcBlock()
        # self.classifier = ClassBlock(input_dim=512, class_num=2)
        self.classifier = Fc_ClassBlock(input_dim=512, class_num=2, dropout=0.75, relu=False)

    def forward(self, d, w, label=None):
        use_gpu = torch.cuda.is_available()
        batch_size = len(d[0])
        num_p_per_id = len(d[0][0])  # 1
        num_g_per_id = len(d[0][0][0])  # 3
        num_p_per_batch = len(d[0]) * len(d[0][0])  # 48
        num_g_per_batch = len(d[0]) * len(d[0][0][0])  # 144
        len_feature = d.shape[-1]
        t = torch.FloatTensor(d.shape).zero_()
        d_new = torch.FloatTensor(d.shape).zero_()
        # this w for dynamic calculate the weight
        # w = torch.FloatTensor(batch_size, batch_size, num_g_per_batch, num_g_per_batch, 1).zero_()
        # this w for calculate the weight by label
        result = torch.FloatTensor(d.shape[: -1] + (2,)).zero_()
        if use_gpu:
            d = d.cuda()
            d_new = d_new.cuda()
            t = t.cuda()
            w = w.cuda()
            result = result.cuda()
            if label is not None:
                label = label.cuda()

        # print('batch_size = %d  num_p_per_batch = %d  num_g_per_batch = %d' % (batch_size, num_p_per_batch, num_g_per_batch))
        for k in range(batch_size):
            for i in range(num_p_per_id):
                for j in range(num_g_per_id):
                    t[k, :, i, j] = self.rf(d[k, :, i, j])

        d = d.reshape(batch_size, -1, len_feature)
        d_new = d_new.reshape(batch_size, -1, len_feature)
        t = t.reshape(d.shape)
        w = w.reshape(batch_size * num_g_per_id, -1)
        result = result.reshape(batch_size, -1, 2)
        if label is not None:
            label = label.reshape(batch_size, -1)

        # w need to be normalized
        w = self.preprocess_adj(w)
        for i in range(t.shape[-1]):
            d_new[:, :, i] = torch.mm(t[:, :, i], w)

        # maybe need to fix
        for i in range(num_p_per_batch):
            # feature = self.fc(d_new[i, :])
            # feature = self.classifier(feature)
            feature = self.classifier.classifier(d_new[i, :])
            result[i, :] = feature.squeeze()

        result = result.view((num_p_per_batch * num_g_per_batch), -1)
        if label is not None:
            label = label.view(label.size(0) * label.size(1))

        print('run Sggnn_gcn foward success  !!!')
        if label is not None:
            return result, label
        else:
            return result

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def preprocess_adj_np(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)
