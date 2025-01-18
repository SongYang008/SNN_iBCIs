# -*- coding: utf-8 -*-

import os
import time

import torch
from spikingjelly.clock_driven.functional import reset_net
from torch.utils.data import DataLoader

from utils.EarlyStop import EarlyStop


def train_test_separate(model, args, file, train_set, val_set, test_sets, et=0, session_id=0):
    print('experiment number %d' % et)
    file.write('experiment number %d' % et)
    file.write('\n')
    file.flush()
    model = model.cuda()
    model.train()
    # train_set, val_set, test_set = LOOSet(root=args.root, session=session)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    clock = 0.
    ckp = args.ckp
    os.makedirs(ckp, exist_ok=True)
    # save_path = ('%s%s%s_s%dexp%d%sg%dc%d%d%s.pth' %
    #              (ckp, args.backbone,
    #               args.task_type, session_id, et, args.spn, args.g, args.cout, args.ts, 'ag' if args.ag else ''))
    arg_dict = vars(args)
    file_name = ''
    for k in arg_dict.keys():
        if k == 'root' or k == 'ckp' or k == 'log':
            continue
        file_name = file_name + str(k) + '_' + str(arg_dict[k])
    save_path = '%s%ss_%de_%d.pth' % (ckp, file_name, session_id, et)
    es = EarlyStop(model=model, patience=100,
                   save_path=save_path)
    for e in range(args.epoch):
        model.train()
        t_bg = time.time()
        loss_add = torch.tensor(0.).cuda()
        ttt = 0.
        tcr = 0.
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            try:
                if args.FF:
                    a, b, out = model(x)
                    # loss = 1 - torch.mean(torch.abs(torch.cosine_similarity(a, b, dim=-1))) + criterion(out, y)
                    loss = criterion(out, y)
                else:
                    out = model(x)
                    loss = criterion(out, y)
            except Exception as e:
                print(e)
                out = model(x)
                loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_add += loss.item()
            preds = torch.argmax(out, dim=1)
            tcr += (preds == y).sum()
            ttt += out.size(0)
            reset_net(model)

        t_ed = time.time()
        clock += (t_ed - t_bg)
        # validate to early stop
        vtt = 0.
        vcr = 0.
        for xv, yv in val_loader:
            model.eval()
            xv, yv = xv.cuda(), yv.cuda()
            try:
                if args.FF:
                    a, b, out = model(xv)
                    # loss = torch.mean(torch.abs(torch.cosine_similarity(a, b, dim=-1))) + criterion(out, y)
                else:
                    out = model(xv)
            except:
                out = model(xv)
            pred = torch.argmax(out, dim=1)
            vtt += out.size(0)
            vcr += (pred == yv).sum()
            reset_net(model)

        vacc = vcr / vtt
        es(vacc)

        if es.stop:
            file.write('early stop at epoch: %d\n' % (e + 1))
            print('early stop at epoch %d' % (e + 1))
            break
        elif e == args.epoch - 1:
            file.write('no early stop\n')
            print('no early stop')

    model.eval()
    model.load_state_dict(torch.load(save_path, weights_only=True))
    accs = []
    if type(test_sets) is not list:
        test_sets = [test_sets]
    for ti, test_set in enumerate(test_sets):
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
        tt = 0.
        cr = 0.
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            try:
                if args.FF:
                    a, b, out = model(x)
                    # loss = torch.mean(torch.abs(torch.cosine_similarity(a, b, dim=-1))) + criterion(out, y)
                else:
                    out = model(x)
            except:
                out = model(x)
            preds = torch.argmax(out, dim=1)
            cr += (preds == y).sum()
            tt += out.size(0)
            reset_net(model)

        acc = cr / tt
        accs.append(acc.cpu().numpy())
    temp = 'experiment number %d, val acc: %.4f test acc:' % (et, es.acc)
    for ac in accs:
        temp = temp + ' %.4f' % ac
    file.write(temp + '\n')
    print(temp)
    # file.write('\n')
    return accs
