import os
import numpy as np

value = [[0, 100], [70, 90], [60, 80], [60, 90], [50, 100], [80, 90], [70, 80], [80, 80], [90, 90], ]
for i in np.arange(3):
    print('i = %.3f' % i)
    log_name = 'log/' + 'log_' + str(i)
    print('log name = %s' % log_name)
    cmd = 'python train.py --net_loss_model ' + str(
        i) + ' --use_dense --gpu_ids 0 --name ft_DesNet --train_all --batchsize  24  --erasing_p 0.5' + ' >> ' + log_name
    print('cmd = %s' % cmd)
    os.system(cmd)

    os.system('python test.py  --use_dense --gpu_ids 0 --name ft_DesNet ' + ' >>  ' + log_name)
    os.system('python evaluate_gpu.py' + ' >> ' + log_name)
    os.system('python evaluate_rerank.py' + ' >> ' + log_name)
