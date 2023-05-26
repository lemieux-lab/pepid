import pepid_utils
import blackboard

blackboard.config.read(blackboard.here("data/default.cfg"))

import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy
import tables

import tqdm

import os
import sys

CUDA = True

BATCH_SIZE = 16
MAX_SIZE = 50000

INIT_PATIENCE = 999

raw_data = tables.open_file("data/massive.h5", 'r')

from specgen import *

class MsDataset(Dataset):
    def __init__(self, data, idxs, max_charge=MAX_CHARGE, epoch_len=None, theoretical=True, generated=False, shuffle=False):
        self.data = data
        self.idxs = idxs
        numpy.random.shuffle(self.idxs)
        self.theoretical = theoretical or generated
        self.generated = generated
        self.shuffle = shuffle
        self.len = epoch_len if epoch_len is not None else len(idxs)
        self.max_charge = max_charge

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        seq = None
        while seq is None or (len(seq) >= PROT_STR_LEN):
            idx = numpy.random.choice(len(self.idxs))
            rawseq_ = self.data.root.meta[self.idxs[idx]]['seq'].decode('utf-8')
            if rawseq_[0] in '-+':
                continue
            seq = ""
            mods = []
            modmode = False
            modstr = ""
            rawseq = rawseq_.replace("_", "").replace("M(ox)", "M+15.995").replace("C+", "X").replace("C", "C+57.021").replace("X", "C+")
            for c in rawseq:
                if not modmode:
                    if re.match("[A-Z]", c) is not None:
                        mods.append(0)
                        seq = seq + c
                    else:
                        modmode = True
                        modstr = modstr + c
                else:
                    if re.match("[A-Z]", c) is not None:
                        seq = seq + c
                        mods[-1] = float(modstr)
                        modstr = ""
                        mods.append(0)
                        modmode = False
                    else:
                        modstr += c
            if modmode:
                mods[-1] = float(modstr)
        
            charge = self.data.root.meta[self.idxs[idx]]['charge']
            if charge > self.max_charge:
                seq = None
                continue

        spec = self.data.root.spectrum[self.idxs[idx]]
        th_spec = make_input(seq, mods)
        spec_fwd = prepare_spec(spec)

        enc_seq = th_spec
          
        out_set = []
        out_set.append(torch.FloatTensor(enc_seq))
        out_set.append(torch.LongTensor([charge-1])) # 0-index the charge...
        out_set.append(torch.FloatTensor([mass]))
        out_set.append(torch.FloatTensor(spec_fwd))
        return tuple(out_set)

def make_run():
    model = Model()

    if CUDA:
        torch.backends.cudnn.enabled = True
        model.cuda()

    print(model)
    #model.load_state_dict(torch.load("best_specgen.pkl"))

    numpy.random.seed(0)

    all_idxs = numpy.arange(len(raw_data.root.meta))
    unique_peps = numpy.unique(list(map(lambda x: x.decode("utf-8").replace("_", "").replace("M(ox)", "1"), raw_data.root.meta[all_idxs]['seq'])))
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

    dataset = MsDataset(raw_data, idxs, epoch_len=None)
    dataset_test = MsDataset(raw_data, test_idxs, epoch_len=int(MAX_SIZE*0.1))
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    dl_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    loss_fn = torch.nn.CosineSimilarity(dim=-1).cuda() #torch.nn.MSELoss().cuda()
    #loss_fn = torch.nn.MSELoss().cuda()
    nll_loss = torch.nn.NLLLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    patience = INIT_PATIENCE
    best_loss = float('inf')
    last_epoch = 200
    epoch_idx = 0

    prev_sim = 0

    while epoch_idx < last_epoch and patience > 0:
        loss_value = {'train': 0, 'test': 0}
        cos_sim = {'train': 0, 'test': 0}
        sparsity = {'train': 0, 'test': 0}
        mass_delta = {'train': 0, 'test': 0}
        charge_acc = {'train': 0, 'test': 0}

        for this_dl in (dl, dl_test):
            phase = 'train' if this_dl is dl else 'test'
            if this_dl is dl:
                model.train()
            else:
                model.eval()
            n = 0
            bar = tqdm.tqdm(enumerate(this_dl), total=len(this_dl))
            for it, batch in bar:
                if this_dl is dl:
                    optimizer.zero_grad()
                seq, charge, mass, spec = batch
                seq = seq.cuda()
                spec = spec.cuda()
                mass = mass.view(-1).cuda()
                charge = charge.view(-1).long().cuda()

                spec = spec.view(-1, spec.shape[-1])
                if this_dl is not dl:
                    with torch.no_grad():
                        pred = model(seq)
                else:
                    pred = model(seq)

                pred_select = torch.stack([pred[i,charge[i],:] for i in range(pred.shape[0])])
                pred_select_sparse = pred_select * (pred_select > 1e-3)
                #l_expr = -loss_fn(pred_select_sparse, spec[:,:PROT_TGT_LEN]).mean() - loss_fn(pred_select, spec[:,:PROT_TGT_LEN]).mean()
                l_expr = -loss_fn(pred_select_sparse, spec[:,:PROT_TGT_LEN]).mean() #+ nll_loss(pred_charge, charge) + ((pred_mass - (mass/5000))**2).mean()

                loss_value[phase] += l_expr.data.cpu().numpy()
                spec_post = spec[:,:PROT_TGT_LEN] / (torch.norm(spec[:,:PROT_TGT_LEN], p=2, dim=-1, keepdim=True) + 1e-10)
                pred_post = pred_select / (torch.norm(pred_select, p=2, dim=-1, keepdim=True) + 1e-10)
                cos_sim[phase] += (spec_post * pred_post).sum(dim=-1).mean().detach().cpu().numpy()
                #charge_acc[phase] += (pred_charge.argmax(dim=-1) == charge).float().mean()
                #mass_delta[phase] += torch.abs(pred_mass - (mass/5000)).mean()

                if this_dl is dl:
                    l_expr.backward(retain_graph=False)
                    optimizer.step()

                sparsity[phase] += (pred_select_sparse / (pred_select_sparse + 1e-10)).mean()

                n += 1

                bar.set_description("[{}] {}: {:.3f} -> L:{:.3f} S:{:.3f} C:{:.3f} M:{:.3f}".format(epoch_idx, "T{}ing".format(phase[1:]), cos_sim[phase] / max(n, 1), loss_value[phase] / max(n, 1), sparsity[phase] / max(n, 1), charge_acc[phase] / max(n, 1), mass_delta[phase] / max(n, 1)))

            loss_value[phase] /= n
            cos_sim[phase] /= n
            sparsity[phase] /= n
            mass_delta[phase] /= n
            charge_acc[phase] /= n

        prev_sim = cos_sim['test']

        if loss_value['test'] < best_loss:
            best_loss = loss_value['test']
            model.save("best_specgen.pkl")
            patience = INIT_PATIENCE
        else:
            patience -= 1
 
        epoch_idx += 1

if __name__ == '__main__':
    make_run()
