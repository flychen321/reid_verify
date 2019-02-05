# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy
from PIL import Image
import time
import os
# from reid_sampler import StratifiedSampler
from model import ft_net, ft_net_dense, PCB, verif_net
from model import Sggnn_siamese, Sggnn_gcn, SiameseNet
from random_erasing import RandomErasing
from datasets import TripletFolder, SiameseDataset, SggDataset
import yaml
from shutil import copyfile
from losses import ContrastiveLoss, SigmoidLoss

version = torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default='data/market/pytorch', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=48, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50')
parser.add_argument('--net_loss_model', default=0, type=int, help='net_loss_model')

opt = parser.parse_args()
print('net_loss_model = %d' % opt.net_loss_model)
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
# print(gpu_ids[0])


######################################################################
# Load Data
# ---------
#

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.PCB:
    transform_train_list = [
        transforms.Resize((384, 192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform_val_list = [
        transforms.Resize(size=(384, 192), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}

# dataset = TripletFolder
dataset = SiameseDataset
image_datasets['train'] = dataset(os.path.join(data_dir, 'train_all'),
                                  data_transforms['train'])
image_datasets['val'] = dataset(os.path.join(data_dir, 'val'),
                                data_transforms['val'])

dataloaders_gcn = {}
dataloaders_gcn['train'] = torch.utils.data.DataLoader(
    SggDataset(os.path.join(data_dir, 'train_all'), data_transforms['train']), batch_size=opt.batchsize, shuffle=True,
    num_workers=8)

batch = {}

class_names = image_datasets['train'].classes
class_vector = [s[1] for s in image_datasets['train'].samples]
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()

since = time.time()
# inputs, classes, pos, pos_classes = next(iter(dataloaders['train']))
print(time.time() - since)

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model_triplet(model, model_verif, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
    last_margin = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_verif_loss = 0.0
            running_corrects = 0.0
            running_verif_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, pos, neg = data
                now_batch_size, c, h, w = inputs.shape

                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    pos = Variable(pos.cuda())
                    neg = Variable(neg.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs, f = model(inputs)
                _, pf = model(pos)
                _, nf = model(neg)
                # pscore = model_verif(pf * f)
                # nscore = model_verif(nf * f)
                pscore = model_verif((pf - f).pow(2))
                nscore = model_verif((nf - f).pow(2))
                # print(pf.requires_grad)
                # loss
                # ---------------------------------
                labels_0 = torch.zeros(now_batch_size).long()
                labels_1 = torch.ones(now_batch_size).long()
                labels_0 = Variable(labels_0.cuda())
                labels_1 = Variable(labels_1.cuda())

                _, preds = torch.max(outputs.data, 1)
                _, p_preds = torch.max(pscore.data, 1)
                _, n_preds = torch.max(nscore.data, 1)
                loss_id = criterion(outputs, labels)
                loss_verif = (criterion(pscore, labels_0) + criterion(nscore, labels_1)) * 0.5 * opt.alpha
                if opt.net_loss_model == 0:
                    loss = loss_id + loss_verif
                elif opt.net_loss_model == 1:
                    loss = loss_verif
                elif opt.net_loss_model == 2:
                    loss = loss_id
                else:
                    print('opt.net_loss_model = %s    error !!!' % opt.net_loss_model)
                    exit()
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0 and 0.5.0
                    running_loss += loss.item()  # * opt.batchsize
                    running_verif_loss += loss_verif.item()  # * opt.batchsize
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0]
                    running_verif_loss += loss_verif.data[0]
                running_corrects += float(torch.sum(preds == labels.data))
                running_verif_corrects += float(torch.sum(p_preds == 0)) + float(torch.sum(n_preds == 1))

            datasize = dataset_sizes['train'] // opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
            epoch_verif_loss = running_verif_loss / datasize
            epoch_acc = running_corrects / datasize
            epoch_verif_acc = running_verif_corrects / (2 * datasize)

            print('{} Loss: {:.4f} Loss_verif: {:.4f}  Acc: {:.4f} Verif_Acc: {:.4f} '.format(
                phase, epoch_loss, epoch_verif_loss, epoch_acc, epoch_verif_acc))
            # if phase == 'val':
            #     if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
            #         best_acc = epoch_acc
            #         best_loss = epoch_loss
            #         best_epoch = epoch
            #         best_model_wts = model.state_dict()
            #     if epoch >= 0:
            #         save_network(model, epoch)

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            epoch_acc = (epoch_acc + epoch_verif_acc) / 2.0
            epoch_loss = (epoch_loss + epoch_verif_loss) / 2.0
            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_network(model, 'best')

            if epoch % 10 == 9:
                save_network(model, epoch)
            draw_curve(epoch)
            last_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


def train_model_siamese_with_two_model(model, model_verif, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    last_margin = 0.0
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_id_loss = 0.0
            running_verif_loss = 0.0
            running_id_corrects = 0.0
            running_verif_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, vf_labels, id_labels = data
                now_batch_size, c, h, w = inputs[0].shape
                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                if type(inputs) not in (tuple, list):
                    inputs = (inputs,)
                if type(id_labels) not in (tuple, list):
                    id_labels = (id_labels,)
                if use_gpu:
                    inputs = tuple(d.cuda() for d in inputs)
                    id_labels = tuple(d.cuda() for d in id_labels)
                    if vf_labels is not None:
                        vf_labels = vf_labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs1, f1 = model(inputs[0])
                outputs2, f2 = model(inputs[1])
                score = model_verif((f1 - f2).pow(2))
                _, id_preds1 = torch.max(outputs1.data, 1)
                _, id_preds2 = torch.max(outputs2.data, 1)
                _, vf_preds = torch.max(score.data, 1)
                loss_id1 = criterion(outputs1, id_labels[0])
                loss_id2 = criterion(outputs2, id_labels[1])
                loss_id = loss_id1 + loss_id2
                loss_verif = criterion(score, vf_labels)
                # loss = loss_verif * opt.alpha + loss_id
                loss = loss_verif
                # if opt.net_loss_model == 0:
                #     loss = loss_id + loss_verif
                # elif opt.net_loss_model == 1:
                #     loss = loss_verif
                # elif opt.net_loss_model == 2:
                #     loss = loss_id
                # else:
                #     print('opt.net_loss_model = %s    error !!!' % opt.net_loss_model)
                #     exit()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0 and 0.5.0
                    running_id_loss += loss.item()  # * opt.batchsize
                    running_verif_loss += loss_verif.item()  # * opt.batchsize
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_id_loss += loss.data[0]
                    running_verif_loss += loss_verif.data[0]
                running_id_corrects += float(torch.sum(id_preds1 == id_labels[0].data))
                running_id_corrects += float(torch.sum(id_preds2 == id_labels[1].data))
                running_verif_corrects += float(torch.sum(vf_preds == vf_labels))

            datasize = dataset_sizes['train'] // opt.batchsize * opt.batchsize
            epoch_id_loss = running_id_loss / datasize
            epoch_verif_loss = running_verif_loss / datasize
            epoch_id_acc = running_id_corrects / (datasize * 2)
            epoch_verif_acc = running_verif_corrects / datasize

            print('{} Loss_id: {:.4f} Loss_verif: {:.4f}  Acc_id: {:.4f} Verif_Acc: {:.4f} '.format(
                phase, epoch_id_loss, epoch_verif_loss, epoch_id_acc, epoch_verif_acc))

            epoch_acc = (epoch_id_acc + epoch_verif_acc) / 2.0
            epoch_loss = (epoch_id_loss + epoch_verif_loss) / 2.0
            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_network(model, 'best')

            y_loss[phase].append(epoch_id_loss)
            y_err[phase].append(1.0 - epoch_id_acc)
            # deep copy the model

            if epoch % 10 == 9:
                save_network(model, epoch)

            draw_curve(epoch)
            last_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


def train_model_siamese(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    last_margin = 0.0
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_id_loss = 0.0
            running_verif_loss = 0.0
            running_id_corrects = 0.0
            running_verif_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, vf_labels, id_labels = data
                now_batch_size, c, h, w = inputs[0].shape
                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                if type(inputs) not in (tuple, list):
                    inputs = (inputs,)
                if type(id_labels) not in (tuple, list):
                    id_labels = (id_labels,)
                if use_gpu:
                    inputs = tuple(d.cuda() for d in inputs)
                    id_labels = tuple(d.cuda() for d in id_labels)
                    if vf_labels is not None:
                        vf_labels = vf_labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs1, f1, outputs2, f2, feature, score = model(inputs[0], inputs[1])
                _, id_preds1 = torch.max(outputs1.data, 1)
                _, id_preds2 = torch.max(outputs2.data, 1)
                _, vf_preds = torch.max(score.data, 1)
                loss_id1 = criterion(outputs1, id_labels[0])
                loss_id2 = criterion(outputs2, id_labels[1])
                loss_id = loss_id1 + loss_id2
                loss_verif = criterion(score, vf_labels)
                # loss = loss_verif * opt.alpha + loss_id
                loss = loss_verif
                # if opt.net_loss_model == 0:
                #     loss = loss_id + loss_verif
                # elif opt.net_loss_model == 1:
                #     loss = loss_verif
                # elif opt.net_loss_model == 2:
                #     loss = loss_id
                # else:
                #     print('opt.net_loss_model = %s    error !!!' % opt.net_loss_model)
                #     exit()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0 and 0.5.0
                    running_id_loss += loss.item()  # * opt.batchsize
                    running_verif_loss += loss_verif.item()  # * opt.batchsize
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_id_loss += loss.data[0]
                    running_verif_loss += loss_verif.data[0]
                running_id_corrects += float(torch.sum(id_preds1 == id_labels[0].data))
                running_id_corrects += float(torch.sum(id_preds2 == id_labels[1].data))
                running_verif_corrects += float(torch.sum(vf_preds == vf_labels))

            datasize = dataset_sizes['train'] // opt.batchsize * opt.batchsize
            epoch_id_loss = running_id_loss / datasize
            epoch_verif_loss = running_verif_loss / datasize
            epoch_id_acc = running_id_corrects / (datasize * 2)
            epoch_verif_acc = running_verif_corrects / datasize

            print('{} Loss_id: {:.4f} Loss_verif: {:.4f}  Acc_id: {:.4f} Verif_Acc: {:.4f} '.format(
                phase, epoch_id_loss, epoch_verif_loss, epoch_id_acc, epoch_verif_acc))

            epoch_acc = (epoch_id_acc + epoch_verif_acc) / 2.0
            epoch_loss = (epoch_id_loss + epoch_verif_loss) / 2.0
            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_network(model, 'best')

            y_loss[phase].append(epoch_id_loss)
            y_err[phase].append(1.0 - epoch_id_acc)
            # deep copy the model

            if epoch % 10 == 9:
                save_network(model, epoch)

            draw_curve(epoch)
            last_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


def train_gcn(train_loader, model_siamese, loss_siamese_fn, optimizer_siamese, scheduler_siamese,
                      model_gcn, loss_gcn_fn, optimizer_gcn, scheduler_gcn, num_epochs=25):
    global cnt
    since = time.time()
    model_gcn.train(True)
    model_siamese.train(False)
    losses = []
    total_loss = 0
    for epoch in range(num_epochs):
        scheduler_siamese.step()
        scheduler_gcn.step()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for batch_idx, (data, target) in enumerate(train_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if use_gpu:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            optimizer_gcn.zero_grad()

            with torch.no_grad():
                outputs = model_siamese(*data, target)

            outputs, target = model_gcn(*outputs)  # for SGGNN

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_gcn_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer_gcn.step()
            print('batch_idx = %4d  loss = %f' % (batch_idx, loss))

    time_elapsed = time.time() - since
    print('time = %f' % (time_elapsed))
    save_network(model_gcn, 'best')
    return model_gcn


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="triplet_loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    #    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    #    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))


######################################################################
# Load model
# ---------------------------
def load_network_easy(network):
    save_path = os.path.join('./model', name, 'net_%s.pth' % 'last')
    print('load pretrained model: %s' % save_path)
    network.load_state_dict(torch.load(save_path))
    return network


def load_network(network, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'best')
    else:
        save_path = model_name
    print('load pretrained model: %s' % save_path)
    net_original = torch.load(save_path)
    pretrained_dict = net_original.state_dict()
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network


######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.state_dict(), save_path)
    # if torch.cuda.is_available:
    #     network.cuda(gpu_ids[0])


def save_whole_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network, save_path)


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

if opt.use_dense:
    model = ft_net_dense(len(class_names))
else:
    model = ft_net(len(class_names))

if opt.PCB:
    model = PCB(len(class_names))

model_verif = verif_net()
# print(model)
# print(model_verif)

if use_gpu:
    model = model.cuda()
    model_verif = model_verif.cuda()

criterion = nn.CrossEntropyLoss()

if not opt.PCB:
    ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.model.fc.parameters(), 'lr': opt.lr},
        {'params': model.classifier.parameters(), 'lr': opt.lr},
        {'params': model_verif.classifier.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    ignored_params = list(map(id, model.model.fc.parameters()))
    ignored_params += (list(map(id, model.classifier0.parameters()))
                       + list(map(id, model.classifier1.parameters()))
                       + list(map(id, model.classifier2.parameters()))
                       + list(map(id, model.classifier3.parameters()))
                       + list(map(id, model.classifier4.parameters()))
                       + list(map(id, model.classifier5.parameters()))
                       # +list(map(id, model.classifier6.parameters() ))
                       # +list(map(id, model.classifier7.parameters() ))
                       )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.001},
        {'params': model.model.fc.parameters(), 'lr': 0.01},
        {'params': model.classifier0.parameters(), 'lr': 0.01},
        {'params': model.classifier1.parameters(), 'lr': 0.01},
        {'params': model.classifier2.parameters(), 'lr': 0.01},
        {'params': model.classifier3.parameters(), 'lr': 0.01},
        {'params': model.classifier4.parameters(), 'lr': 0.01},
        {'params': model.classifier5.parameters(), 'lr': 0.01},
        # {'params': model.classifier6.parameters(), 'lr': 0.01},
        # {'params': model.classifier7.parameters(), 'lr': 0.01}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[40, 60], gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
    # copyfile('./train.py', dir_name + '/train.py')
    # copyfile('./model.py', dir_name + '/model.py')
    # copyfile('./datasets.py', dir_name + '/datasets.py')

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

stage_0 = False
stage_1 = False
stage_2 = True

# if stage_0:
#     # train_model = train_model_triplet
#     train_model = train_model_siamese_with_two_model
#     model = train_model(model, model_verif, criterion, optimizer_ft, exp_lr_scheduler,
#                         num_epochs=60)

if stage_1:
    embedding_net = ft_net_dense(len(class_names))
    model_siamese = SiameseNet(embedding_net)
    if use_gpu:
        model_siamese.cuda()
    print('model_siamese structure')
    print(model_siamese)

    stage_1_id = list(map(id, model_siamese.parameters()))
    stage_1_base_id = list(map(id, model_siamese.embedding_net.parameters()))
    stage_1_base_params = filter(lambda p: id(p) in stage_1_base_id, model_siamese.parameters())
    stage_1_classifier_params = filter(lambda p: id(p) in stage_1_id and id(p) not in stage_1_base_id,
                                       model_siamese.parameters())

    optimizer_ft = optim.Adam([
        {'params': stage_1_base_params, 'lr': 0.1 * opt.lr},
        {'params': stage_1_classifier_params, 'lr': 1 * opt.lr},
    ])

    # optimizer_ft = optim.SGD([
    #     {'params': model_siamese.parameters(), 'lr': opt.lr}
    # ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[40, 60], gamma=0.1)
    model = train_model_siamese(model_siamese, criterion, optimizer_ft, exp_lr_scheduler,
                                num_epochs=60)

if stage_2:
    margin = 1.
    embedding_net = ft_net_dense(len(class_names))
    model_siamese = Sggnn_siamese(SiameseNet(embedding_net))
    model_gcn = Sggnn_gcn()

    if use_gpu:
        model_siamese.cuda()
        model_gcn.cuda()
    # model_siamese = load_network(model_siamese)
    # loss_siamese_fn = ContrastiveLoss(margin)
    loss_siamese_fn = nn.CrossEntropyLoss()
    loss_gcn_fn = nn.CrossEntropyLoss()
    lr = 1e-3
    optimizer_siamese = optim.Adam(model_siamese.parameters(), lr=lr)
    scheduler_siamese = lr_scheduler.StepLR(optimizer_siamese, 8, gamma=0.1, last_epoch=-1)
    optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=lr)
    scheduler_gcn = lr_scheduler.StepLR(optimizer_gcn, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 100
    model = train_gcn(dataloaders_gcn['train'], model_siamese, loss_siamese_fn, optimizer_siamese, scheduler_siamese,
                      model_gcn, loss_gcn_fn, optimizer_gcn, scheduler_gcn, num_epochs=n_epochs)
