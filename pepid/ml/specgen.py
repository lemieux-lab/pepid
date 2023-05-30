import pepid_utils
import blackboard

blackboard.setup_constants()

config = blackboard.config

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy
import numba
import tables

import tqdm

import os
import sys

NTERM, CTERM = config['processing.db'].getfloat('nterm cleavage'), config['processing.db'].getfloat('cterm cleavage')

SIZE_RESOLUTION_FACTOR = 10
MAX_MZ = 5000
PROT_TGT_LEN = MAX_MZ * SIZE_RESOLUTION_FACTOR
MAX_PEAKS = 2000

AA_LIST = pepid_utils.AMINOS
SEQ_SIZE = len(AA_LIST)

PROT_STR_LEN = 40
MAX_CHARGE = 5
N_MASS_REPS = MAX_CHARGE

INP_SIZE = SEQ_SIZE+2

def make_inputs(seqs, seqmods):
    th_spec = numpy.zeros((len(seqs), PROT_TGT_LEN, 5+1), dtype='float32')
    for i, (seq, mods) in enumerate(zip(seqs, seqmods)):
        all_masses = []

        for z in range(1, 6):
            masses = numpy.asarray(pepid_utils.theoretical_masses(seq, mods, nterm=NTERM, cterm=CTERM, exclude_end=True), dtype='float32').reshape((-1,2))
            th_spec[i,:,z-1] = pepid_utils.blit_spectrum(masses, PROT_TGT_LEN, 1.0 / SIZE_RESOLUTION_FACTOR)

        mass = pepid_utils.neutral_mass(seq, mods, nterm=NTERM, cterm=CTERM, z=1)
        th_spec[i,min(PROT_TGT_LEN-1, int(numpy.round(mass * SIZE_RESOLUTION_FACTOR))),5] = 1

    return th_spec

def make_input(seq, mods):
    return make_inputs([seq], [mods])[0]

@numba.njit()
def prepare_spec(spec):
    spec_fwd = pepid_utils.blit_spectrum(spec, PROT_TGT_LEN, 1. / SIZE_RESOLUTION_FACTOR)
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
