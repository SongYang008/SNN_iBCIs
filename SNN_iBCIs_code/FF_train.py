# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np
import torch

from BackBone.Baseline import EEGNet, DeepConvNet, ShallowConvNet, SVM, EEGConFormer, Deformer
from BackBone.FeatureFusion import FF, FFwoP, FFwoPn, FFwoPd
from BackBone.SNN import Net
from train_test import train_test_separate
from utils.functional import seed_all
from utils.load_data import load_all, LOO

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/mnt/data2/songyang/IBCI/data/spike3')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epoch', default=800)
parser.add_argument('--backbone', default='Net')
parser.add_argument('--task_type', default='direction')
parser.add_argument('--g', default=10, type=int, help='groups of conv')
parser.add_argument('--spn', default='PLIF', type=str, help='Spike neuron types')
parser.add_argument('--ckp', default='./fannckpoint/', type=str, help='checkpoint')
parser.add_argument('--cout', default=48)
parser.add_argument('--ag', action='store_true', dest='ag')
parser.set_defaults(ag=False, type=bool)
parser.set_defaults(os=False, type=bool)
parser.add_argument('--FANN', default=True)
parser.add_argument('--P', type=str, default='0')
parser.add_argument('--d', default=128, type=int)
args = parser.parse_args()

net_dict = {'EEGNet': EEGNet, 'ShallowConvNet': ShallowConvNet, 'SVM': SVM,
            'ConFormer': EEGConFormer, 'DeepConvNet': DeepConvNet,
            'Net': Net, 'Deformer': Deformer, }

session_dict = {'direction': 14, 'shape': 12, }
step_dict = {'3': 100, '2': 250, '1': 500, 'e': 8}

if __name__ == '__main__':
    t_bg = time.time()
    if args.task_type == 'shape':
        in_ch = 66
        args.log = './fannlog1/'
    else:
        args.log = './fannlog/'
        in_ch = 80
    os.makedirs(args.log, exist_ok=True)
    time_step = step_dict[args.root[-1]]
    args.ts = time_step
    num_sessions = session_dict[args.task_type]

    file_name = ''
    arg_dict = vars(args)
    for k in arg_dict.keys():
        if k == 'root' or k == 'ckp' or k == 'log' or k == 'type':
            continue
        file_name = file_name + str(k) + '_' + str(arg_dict[k])
    file_path = ('%s%s.txt' %
                 (args.log, file_name))
    file = open(file_path, 'w')
    datasets = load_all(args.root)
    seeds = [2024, 2025, 2026, 2023, 3407]

    session_acc = []
    for ses in range(num_sessions):
        file.write('*' * 20 + 'session %d' % ses + '*' * 20 + '\n')
        accs = []
        for seed_idx, seed in enumerate(seeds):
            seed_all(seed)
            train_set, val_set, test_set = LOO(datasets, ses)
            torch.set_num_threads(4)
            len_test = 0
            if args.P == '0':
                model = FFwoP(in_channels=in_ch, time_step=time_step, model=net_dict[args.backbone])
            elif args.P == '-1':
                model = FF(in_channels=in_ch, time_step=time_step, model=net_dict[args.backbone])
            elif args.P == '1':
                model = FFwoPn(in_channels=in_ch, time_step=time_step, model=net_dict[args.backbone], d=args.d)
            else:
                model = FFwoPd(in_channels=in_ch, time_step=time_step, model=net_dict[args.backbone], d=args.d)
            for n, p in model.named_parameters():
                if p.dim() >= 2:
                    torch.nn.init.kaiming_uniform_(p, 0)
                    # torch.nn.init.kaiming_normal_(p, 0)

            acc_et = train_test_separate(
                model=model,
                args=args,
                file=file,
                train_set=train_set,
                val_set=val_set,
                test_sets=test_set,
                et=seed_idx,
                session_id=ses
            )
            accs.append(acc_et)
        file.write('\n')
        accs = np.array(accs)
        mean_acc = np.mean(accs, axis=0)[0]
        std_acc = np.std(accs, axis=0)[0]
        # all_test_acc = np.mean(mean_acc)
        print('session %d, result: acc %.4f, std: %.4f' % (ses, mean_acc, std_acc))
        file.write('session %d result: acc %.4f, std: %.4f\n\n' % (ses, mean_acc, std_acc))
        session_acc.append(mean_acc)
        file.flush()
        # file.close()
        if args.os:
            break
    t_ed = time.time()
    spent = t_ed - t_bg
    hours = spent // (60 * 60)
    minutes = spent % 3600 // 60
    seconds = spent % 3600 % 60
    print('whole training process spent %d hours, %d minutes %.f seconds' % (hours, minutes, seconds))
    file.write('whole training process spent %d hours, %d minutes %.f seconds\n' % (hours, minutes, seconds))
    temp = 'all acc:'
    for sa in session_acc:
        temp = temp + ' ' + str(sa)
    file.write(temp + '\n')
    print(temp)
    result_acc = np.mean(session_acc)
    result_std = np.std(session_acc)
    file.write('result acc: %f, std: %f\n' % (float(result_acc), float(result_std)))
    print('result acc: %f, std: %f' % (float(result_acc), float(result_std)))
    file.close()
