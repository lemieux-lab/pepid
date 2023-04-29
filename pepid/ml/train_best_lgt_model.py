import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy
import tables
import time
import json
import pickle

import os
import sys
import itertools

import time

from best_lgt_model import *
import tqdm

MAX_EPOCHS = 10000
def make_run(data, outdir, args):
    numpy.random.seed(0)

    raw_data = tables.open_file("../data/massive.h5", 'r')
    #print(raw_data.root.meta[:]['mass'].max())

    #all_idxs = raw_data.root.meta.get_where_list("(score > {}) & (exists > 0.5)".format(SCORE_CUTOFF))
    #all_idxs = raw_data.root.meta.get_where_list("(exists > 0.5)")
    all_idxs = numpy.arange(raw_data.root.meta.shape[0])
    unique_peps = numpy.unique(list(map(lambda x: x.decode('utf-8').replace("_", "").replace("M(ox)", "1"), raw_data.root.meta[all_idxs]['seq'])))
    n_data = unique_peps.shape[0]
    all_pep_idxs = numpy.arange(n_data)
    numpy.random.shuffle(all_pep_idxs)

    train_peptides = all_pep_idxs[:int(0.8*len(all_pep_idxs))]
    test_peptides = all_pep_idxs[int(0.8*len(all_pep_idxs)):int(0.9*len(all_pep_idxs))]
    valid_peptides = all_pep_idxs[int(0.9*len(all_pep_idxs)):]

    all_n = min(all_idxs.shape[0], MAX_SIZE)

    print("n_data: {}".format(n_data))

    raw_data_seqs = list(map(lambda x: x.decode('utf-8').replace("_", "").replace("M(ox)", "1"), raw_data.root.meta[:]['seq']))

    idxs = numpy.argwhere(numpy.isin(raw_data_seqs, unique_peps[train_peptides]))[:,0]
    numpy.random.shuffle(idxs)
    idxs = idxs[:int(all_n * 0.8)]
    test_idxs = numpy.argwhere(numpy.isin(raw_data_seqs, unique_peps[test_peptides]))[:,0]
    numpy.random.shuffle(test_idxs)
    test_idxs = test_idxs[:int(all_n * 0.1)]
    valid_idxs = numpy.argwhere(numpy.isin(raw_data_seqs, unique_peps[valid_peptides]))[:,0]
    numpy.random.shuffle(valid_idxs)
    valid_idxs = valid_idxs[:int(all_n * 0.1)]

    pep_lengths_tr = list(map(lambda x: len(x.decode('utf-8').replace("_", "").replace("M(ox)", "1")), raw_data.root.meta[idxs]['seq']))
    pep_lengths_va = list(map(lambda x: len(x.decode('utf-8').replace("_", "").replace("M(ox)", "1")), raw_data.root.meta[valid_idxs]['seq']))
    pep_lengths_te = list(map(lambda x: len(x.decode('utf-8').replace("_", "").replace("M(ox)", "1")), raw_data.root.meta[test_idxs]['seq']))
    print("Train set size: {}; Valid set size: {}; Test set size: {}".format(len(idxs), len(valid_idxs), len(test_idxs)))
    print("Train lengths: min {} max {}".format(numpy.min(pep_lengths_tr), numpy.max(pep_lengths_tr)))
    print("Valid lengths: min {} max {}".format(numpy.min(pep_lengths_va), numpy.max(pep_lengths_va)))
    print("Test lengths: min {} max {}".format(numpy.min(pep_lengths_te), numpy.max(pep_lengths_te)))

    dataset = MsDataset(raw_data, idxs, epoch_len=50000, class_balance=False)
    dataset_test = MsDataset(raw_data, test_idxs, epoch_len=5000)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True)
    dl_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True)

    model = Model().cuda()
    #model.load_state_dict(torch.load("best_nice_chop3.pkl"))

    i = 1

    lgt_prior = numpy.zeros(GT_MAX_LGT - GT_MIN_LGT+1)
    n_batches = 0
    eye = numpy.eye(GT_MAX_LGT - GT_MIN_LGT+1)
    n_classes = (GT_MAX_LGT - GT_MIN_LGT + 1)
    for batch in dl:
        lgt_prior += eye[batch[-1].detach().cpu().numpy().reshape((-1,)).astype('int32') - GT_MIN_LGT].mean(axis=0)
        n_batches += 1
    lgt_prior /= n_batches
    lgt_weight = numpy.copy(lgt_prior)
    lgt_weight[lgt_weight > 0] = 1.0 / (n_classes * lgt_weight[lgt_weight > 0])
    print(lgt_prior)
    lgt_prior = torch.FloatTensor(numpy.log(numpy.maximum(1e-10, lgt_prior))).view(1, -1).repeat(BATCH_SIZE, 1).cuda()

    #loss_fn = torch.nn.NLLLoss().cuda()
    #loss_fn = torch.nn.BCEWithLogitsLoss().cuda()
    loss_fn = torch.nn.NLLLoss().cuda()
    weighted_loss_fn = torch.nn.NLLLoss(weight=torch.FloatTensor(lgt_weight)).cuda()
    #mse_loss = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float('inf')

    lgt_cnt = numpy.array([0]*34)

    while i < MAX_EPOCHS:
        acc = {'train': 0, 'test': 0}
        acc_rank = {'train': 0, 'test': 0}
        conv = {'train': 0, 'test': 0}
        lin = {'train': 0, 'test': 0}
        meta = {'train': 0, 'test': 0}
        loss = {'train': 0, 'test': 0}
        n = {'train': 0, 'test': 0}

        for this_dl, phase in zip([dl, dl_test], ['train', 'test']):
            bar = tqdm.tqdm(enumerate(this_dl), total=len(this_dl), desc=phase)
            start_time = time.strftime("%I:%M:%S %p", time.localtime())
            for it, batch in bar:
                spec, precmass, lgts = batch
                spec = spec.cuda()
                lgts = lgts.cuda()
                precmass = precmass.cuda()

                #lgt_cnt[lgts.cpu().detach().numpy().astype('int32') - OFFSET] += 1

                if phase == 'train':
                    optimizer.zero_grad()

                ret = model(spec.view(-1, PROT_TGT_LEN), precmass)
                lgts = lgts - OFFSET

                loss_expr = weighted_loss_fn(ret['pred'].view(-1, PROT_STR_LEN - OFFSET + 1), lgts.view(-1).long())
                pre_loss_expr = loss_fn(ret['conv'].view(-1, PROT_STR_LEN - OFFSET + 1), lgts.view(-1).long())
                wloss_expr = weighted_loss_fn(ret['fc'].view(-1, PROT_STR_LEN - OFFSET + 1), lgts.view(-1).long())
                mloss_expr = loss_fn(ret['mass'].view(-1, PROT_STR_LEN - OFFSET + 1), lgts.view(-1).long())
                loss[phase] += loss_expr.data.cpu().numpy()
                acc[phase] += (ret['pred'].argmax(dim=-1).view(-1) == lgts.view(-1)).float().mean().detach().cpu().numpy()
                conv[phase] += (ret['conv'].argmax(dim=-1).view(-1) == lgts.view(-1)).float().mean().detach().cpu().numpy()
                lin[phase] += (ret['fc'].argmax(dim=-1).view(-1) == lgts.view(-1)).float().mean().detach().cpu().numpy()
                meta[phase] += (ret['mass'].argmax(dim=-1).view(-1) == lgts.view(-1)).float().mean().detach().cpu().numpy()
                #adhoc[phase] += (out.argmax(dim=-1).view(-1) - lgts.view(-1)).abs().float().mean().detach().cpu().numpy()
                n[phase] += 1

                if phase == 'train':
                    (pre_loss_expr + loss_expr + wloss_expr + mloss_expr).backward()
                    #pre_loss_expr.backward()
                    optimizer.step()

                bar.set_description("[{}] Epoch {} ({}): ".format(start_time, i, phase) + \
                                "loss: {:.3f} (acc: {:.3f}; conv: {:.3f}, lin: {:.3f}, meta: {:.3f})".format(loss[phase] / max(1, n[phase]), acc[phase] / max(1, n[phase]), conv[phase] / max(1, n[phase]), lin[phase] / max(1, n[phase]), meta[phase] / max(1, n[phase])))

        if loss['test'] / n['test'] < best:
            best = loss['test'] / n['test']
            model.save("best_lgt_model.pkl")
        model.save("latest_lgt_model.pkl")

        i += 1

if __name__ == '__main__':
    make_run(None, None, None)
