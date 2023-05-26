import pepid_utils
import blackboard

blackboard.setup_constants()

config = blackboard.config

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy
import tables

import tqdm

import os
import sys

NTERM, CTERM = config['processing.db'].getfloat('nterm cleavage'), config['processing.db'].getfloat('cterm cleavage')

SIZE_RESOLUTION_FACTOR = 10
MAX_MZ = 5000
PROT_TGT_LEN = MAX_MZ * SIZE_RESOLUTION_FACTOR

AA_LIST = pepid_utils.AMINOS
SEQ_SIZE = len(AA_LIST)

PROT_STR_LEN = 40
MAX_CHARGE = 5
N_MASS_REPS = MAX_CHARGE

INP_SIZE = SEQ_SIZE+2

def make_input(seq, mods):
    all_masses = []
    for z in range(1, 6):
        all_masses.append(numpy.asarray(pepid_utils.theoretical_masses(seq, mods, nterm=NTERM, cterm=CTERM, exclude_end=True)).reshape((-1,2))[:,0])

    th_spec = numpy.zeros((PROT_TGT_LEN, 5+1), dtype='float32')
    for z in range(len(all_masses)):
        for mz in sorted(all_masses[z]):
            if int(numpy.round(mz / SIZE_RESOLUTION_FACTOR)) < PROT_TGT_LEN:
                th_spec[int(numpy.round(mz / SIZE_RESOLUTION_FACTOR)), z] += 1
            else:
                break
        th_spec[:,z] /= (th_spec[:,z].max() + 1e-10)

    mass = pepid_utils.neutral_mass(seq, mods, nterm=NTERM, cterm=CTERM, z=1)
    th_spec[min(PROT_TGT_LEN-1, int(numpy.round(mass / SIZE_RESOLUTION_FACTOR))),5] = 1

    return th_spec

def blit_spec(spec):
    spec_fwd = numpy.zeros((PROT_TGT_LEN,))
    for mz, intens in spec:
        if mz == 0:
            break
        idx = int(numpy.round(mz / SIZE_RESOLUTION_FACTOR))
        if idx < PROT_TGT_LEN:
            spec_fwd[idx] += intens
        else:
            break
    spec_fwd = spec_fwd / (spec_fwd.max() + 1e-10)

    return spec_fwd


def prepare_spec(spec):
    spec_fwd = blit_spec(spec)
    spec_fwd = numpy.sqrt(spec_fwd)

    return spec_fwd

def embed(inp, mass_scale = 5000):
    emb = numpy.zeros((PROT_STR_LEN + 1, INP_SIZE), dtype='float32')

    pep = inp['pep']

    for i in range(len(pep), PROT_STR_LEN): emb[i,pepid_utils.AMINOS.index("_")] = 1. # padding first, meta column should not be affected
    meta = emb[:,SEQ_SIZE:]
    meta[:,0] = pepid_utils.neutral_mass(inp['pep'], inp['mods'], NTERM, CTERM, z=1) / mass_scale
    for i in range(len(pep)):
        emb[i,pepid_utils.AMINOS.index(pep[i])] = 1.
        emb[i,SEQ_SIZE+1] = inp['mods'][i]

    return emb

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.emb_size = 8

        self.enc = nn.Conv1d(5+1+1, self.emb_size, kernel_size=100+1, padding='same')
        self.enc_bn = nn.BatchNorm1d(self.emb_size, momentum=0.01, eps=1e-3, affine=True, track_running_stats=False)
        self.enc_nl = nn.ReLU()

        self.processors = []
        for _ in range(4):
            self.processors.append(nn.Conv1d(self.emb_size, self.emb_size, kernel_size=14+1, padding='same'))
            self.processors.append(nn.BatchNorm1d(self.emb_size, momentum=0.01, eps=1e-3, affine=True, track_running_stats=False))
            self.processors.append(nn.ReLU())
        self.processors = nn.ModuleList(self.processors)

        self.decoder = nn.Conv1d(self.emb_size, 5, kernel_size=100+1, padding='same')
        self.dec_nl = nn.Sigmoid()

    def forward(self, inp):
        inp = torch.cat([inp, torch.arange(inp.shape[1]).to('cuda:0').reshape((1, -1, 1)).repeat(inp.shape[0], 1, 1) / inp.shape[1]], dim=-1)
        inp = inp.transpose(1, 2)
        this_hid = self.enc_nl(self.enc_bn(self.enc(inp))).transpose(1, 2)
        this_hid = this_hid.transpose(1, 2)

        for i, l in enumerate(self.processors):
            this_hid = l(this_hid)

        ret = self.dec_nl(self.decoder(this_hid))

        return ret

    def save(self, fout):
        torch.save(self.state_dict(), open(fout, 'wb'))
