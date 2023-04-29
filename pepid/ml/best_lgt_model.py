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

AMINOS = ['_', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '1'] # '1' = M(ox)
MASSES = [99999., 71.08, 156.2, 114.1, 115.1, 103.1 + 57.02,  128.1, 129.1, 57.05, 137.1, 113.2, 113.2, 128.2, 131.2, 147.2, 97.12, 87.08, 101.1, 186.2, 163.2, 99.07, 165.21]

SIZE_RESOLUTION_FACTOR = 0.1
BATCH_SIZE = 10
MAX_SIZE = 9999999999999
SCORE_CUTOFF = 100.
PROT_STR_LEN = 40
MAX_CHARGE = 5
TOL = 0.1
OFFSET=6
GT_MIN_LGT = 6
GT_MAX_LGT = 40
PROT_BLIT_LEN = (int)(5000 / SIZE_RESOLUTION_FACTOR)
PROT_TGT_LEN = (int)(2000 / SIZE_RESOLUTION_FACTOR)

torch.backends.cudnn.enabled = True

numpy.random.seed(0)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.emb_dim = 500

        self.nl = nn.ReLU()
        self.out_nl = nn.LogSoftmax(dim=1)
        #self.conv0 = nn.Conv1d(1, self.emb_dim, kernel_size=11, padding=0)
        #self.pool0 = nn.AvgPool1d(2)

        kernel = 12
        pool = 2

        self.convs = []
        self.converter = []

        curr_size = float(PROT_TGT_LEN)

        while curr_size != 1:
            next_kernel = max(1, min(kernel, curr_size - 2))
            next_pool = min(curr_size - next_kernel + 1, pool)

            while (curr_size - next_kernel + 1) % next_pool != 0:
                if next_kernel == 1:
                    next_pool = curr_size - next_kernel + 1
                    break
                next_kernel -= 1
                next_pool = min(curr_size - next_kernel + 1, pool)

            old_size = curr_size
            curr_size = (curr_size - next_kernel + 1) / next_pool

            next_pool = int(next_pool)
            next_kernel = int(next_kernel)

            #print("New layer: {} conv {} pool {} -> {}".format(old_size, next_kernel, next_pool, curr_size))

            self.convs.append(nn.Conv1d(1 if len(self.convs) == 0 else self.emb_dim, self.emb_dim if curr_size > 1 else (GT_MAX_LGT-GT_MIN_LGT+1), kernel_size=next_kernel, padding=0))
            if curr_size > 1:
                self.convs.append(nn.ReLU())
            self.convs.append(nn.AvgPool1d(next_pool))
            if curr_size == 1:
                self.convs.append(nn.LogSoftmax(dim=-1))
                self.converter.append(nn.Conv1d(1 if len(self.convs) == 0 else self.emb_dim, self.emb_dim, kernel_size=next_kernel, padding=0))
                self.converter.append(nn.AvgPool1d(next_pool))
                self.converter.append(nn.ReLU())

        self.converter = nn.ModuleList(self.converter)
        self.convs = nn.ModuleList(self.convs)

        self.ldim = 500

        self.lin1 = nn.Linear(PROT_TGT_LEN, self.ldim)
        self.lin2 = nn.Linear(self.ldim, self.ldim)
        self.lin3 = nn.Linear(self.ldim, self.ldim)
        self.lin4 = nn.Linear(self.ldim, self.ldim)
        self.lin5 = nn.Linear(self.ldim, self.ldim)
        self.lin6 = nn.Linear(self.ldim, self.ldim)
        self.lin7 = nn.Linear(self.ldim, self.ldim)
        self.lin8 = nn.Linear(self.ldim, self.ldim)
        self.lin_out = nn.Linear(self.ldim, GT_MAX_LGT-GT_MIN_LGT+1)
        self.sigm = nn.Sigmoid()

        self.lmetadim = 100
        self.linm1 = nn.Linear(1, self.lmetadim)
        self.linm2 = nn.Linear(self.lmetadim, self.lmetadim)
        self.linm3 = nn.Linear(self.lmetadim, self.lmetadim)
        self.linm4 = nn.Linear(self.lmetadim, self.lmetadim)
        self.linm_out = nn.Linear(self.lmetadim, GT_MAX_LGT-GT_MIN_LGT+1)

        self.cl_dim = 100
        self.comb_out = (GT_MAX_LGT - GT_MIN_LGT+1)
        self.comb1 = nn.Linear(self.lmetadim + self.ldim + self.emb_dim, self.cl_dim)
        self.comb2 = nn.Linear(self.cl_dim, self.cl_dim)
        self.comb3 = nn.Linear(self.cl_dim, self.cl_dim)
        self.comb4 = nn.Linear(self.cl_dim, self.cl_dim)
        self.comb_out = nn.Linear(self.cl_dim, GT_MAX_LGT-GT_MIN_LGT+1)

    def forward(self, spec, mass):
        ret = {}

        inp = spec

        wout = self.nl(self.lin1(inp))
        wout = self.nl(self.lin2(wout))
        wout = self.nl(self.lin3(wout))
        wout = self.nl(self.lin4(wout))
        wout = self.nl(self.lin5(wout))
        wout = self.nl(self.lin6(wout))
        wout = self.nl(self.lin7(wout))
        wout = self.nl(self.lin8(wout))
        pre_fc = wout
        ret['fc'] = self.out_nl(self.lin_out(wout))

        inp = inp.view(inp.shape[0],1,-1)
        out = inp
        for layer in self.convs[:-3]:
            out = layer(out)
        pre_conv = out
        out = self.convs[-3](out)
        out = self.convs[-2](out)
        out = out.mean(dim=-1).view(out.shape[0], -1)
        ret['conv'] = self.out_nl(out)

        inp = mass
        mout = self.nl(self.linm1(inp))
        mout = self.nl(self.linm2(mout))
        mout = self.nl(self.linm3(mout))
        mout = self.nl(self.linm4(mout))
        pre_mass = mout
        ret['mass'] = self.out_nl(self.linm_out(mout))

        converted_conv = pre_conv.detach()
        for l in self.converter:
            converted_conv = l(converted_conv)
        converted_conv = converted_conv.mean(dim=-1).view(converted_conv.shape[0], -1)

        out_comb = torch.cat((pre_fc.detach(), converted_conv, pre_mass.detach()), dim=1)
        out_comb = self.nl(self.comb1(out_comb))
        out_comb = self.nl(self.comb2(out_comb))
        out_comb = self.nl(self.comb3(out_comb))
        out_comb = self.nl(self.comb4(out_comb))
        ret['pred'] = self.out_nl(self.comb_out(out_comb))

        return ret

    def save(self, fout):
        torch.save(self.state_dict(), open(fout, 'wb'))

class MsDataset(Dataset):
    def __init__(self, data, idxs, epoch_len=None, shuffle=True, class_balance=False, mean_spectrum=0):
        self.data = data
        self.idxs = idxs
        numpy.random.shuffle(self.idxs)
        self.shuffle = shuffle
        self.len = epoch_len if epoch_len is not None else len(idxs)
        self.class_balance = class_balance
        self.mean_spectrum = mean_spectrum

        self.cache = {}

    def encode_seq(self, seq):
        enc = [AMINOS.index(c) for c in seq]
        enc = enc + [0]
        while True:
            if len(enc) < PROT_STR_LEN:
                enc = enc + [0]
            else:
                break
        return enc

    def decode_seq(self, enc):
        return ''.join([AMINOS[e] if 0 <= e < len(AMINOS) else '-' for e in enc])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        numpy.random.seed()

        lgt = 9999
        lgt_tgt = 9999
        prec_mass = 2001
        charge = 9

        if self.class_balance:
            lgt_tgt = numpy.random.choice(20) + GT_MIN_LGT

        #print(lgt_tgt, self.class_balance)

        import re

        while (not (GT_MIN_LGT <= lgt <= GT_MAX_LGT)):
            idx = int(numpy.random.randint(low=0, high=len(self.idxs), size=(1,)))

            spec_raw = self.data.root.spectrum[self.idxs[idx]]
            spec = numpy.zeros((PROT_TGT_LEN,), dtype='float32')
            for mz, intens in spec_raw:
                if mz * 10 < PROT_TGT_LEN - 0.5:
                    spec[int(numpy.round(mz*10))] += intens
            max = spec.max()
            if max != 0:
                spec /= max
            spec_z = self.data.root.meta[self.idxs[idx]]["charge"]
            spec_mass = self.data.root.meta[self.idxs[idx]]["mass"]
            precmass = spec_mass * spec_z - spec_z * 1.00727646688
            prec_mass = precmass
            charge = spec_z

            seq_raw = self.data.root.meta[self.idxs[idx]]['seq'].decode('utf-8')
            seq = re.sub("[^A-Z]", "", seq_raw.replace("1", "M"))
            lgt = len(seq)

            if not self.class_balance:
                lgt_tgt = lgt

        out_set = []
        out_set.append(torch.FloatTensor(spec))
        out_set.append(torch.FloatTensor([precmass / 2000]))
        out_set.append(torch.FloatTensor([lgt]))

        return tuple(out_set)
