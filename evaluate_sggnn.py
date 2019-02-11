import scipy.io
import torch
import numpy as np
import time
import os
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from model import Sggnn_siamese, Sggnn_gcn, Sggnn_for_test
from model import load_network_easy, load_network, save_network, save_whole_network

######################################################################
# Trained model
print('-------evaluate-----------')
name = 'sggnn'
use_gpu = torch.cuda.is_available()
model_gcn = Sggnn_for_test()
model_gcn = load_network(model_gcn, name, 'whole_best_gcn')
if use_gpu:
    model = model_gcn.cuda()
#######################################################################
# Evaluate

cam_metric = torch.zeros(6, 6)


def evaluate(qf, ql, qc, gf, gl, gc, model=model):
    model.eval()
    query = qf.view(-1, 1)
    # print(query.shape)
    score = ((gf - qf).pow(2)).sum(1)  # Ed distance
    score = score.cpu().numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    # index = index[::-1]      # Ed distance does not need this operate
    # operate for sggnn
    with torch.no_grad():
        index_new_100 = model(qf.unsqueeze(0), gf[index[:100]].unsqueeze(0))
    index[:100] = index[:100][index_new_100.squeeze()]

    # good index
    query_index = np.argwhere(gl == ql)
    # same camera
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, qc, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, qc, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    ranked_camera = gallery_cam[index]
    mask = np.in1d(index, junk_index, invert=True)
    # mask2 = np.in1d(index, np.append(good_index,junk_index), invert=True)
    index = index[mask]
    ranked_camera = ranked_camera[mask]
    for i in range(10):
        cam_metric[qc - 1, ranked_camera[i] - 1] += 1

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
# print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label,
                               gallery_cam)
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    if i % 10 == 0:
        print('i = %3d    CMC_tmp[0] = %s' % (i, CMC_tmp[0].numpy()))

CMC = CMC.float()
CMC = CMC / len(query_label)  # average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
