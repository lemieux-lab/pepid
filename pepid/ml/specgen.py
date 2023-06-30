import pickle
import msgpack

if __package__ == "" or __package__ is None:
    from pepid import pepid_utils
    from pepid import blackboard
else:
    from .. import pepid_utils
    from .. import blackboard

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
MAX_PEAKS = 2000

AA_LIST = pepid_utils.AMINOS
AA_STRING = "".join(AA_LIST)
SEQ_SIZE = len(AA_LIST)

PROT_STR_LEN = 40
MAX_CHARGE = 5
N_MASS_REPS = MAX_CHARGE

INP_SIZE = SEQ_SIZE+2

import numba

@numba.njit()
def prepare_spec(spec):
    spec_fwd = pepid_utils.blit_spectrum(spec, PROT_TGT_LEN, 1.0 / SIZE_RESOLUTION_FACTOR)
    spec_fwd = numpy.sqrt(spec_fwd)
    return spec_fwd

@numba.njit(locals={'i': numba.int32, 'indexed': numba.int32})
def embed(pep, mods, mass, mass_scale = MAX_MZ):
# Input is {"pep": peptide, 'mods': mods, "mass": precursor mass}
    emb = numpy.zeros((PROT_STR_LEN + 1, INP_SIZE), dtype='float32')

    for i in range(len(pep), PROT_STR_LEN, 1):
        indexed = AA_STRING.index("_")
        emb[i][indexed] = 1. # padding first, meta column should not be affected
    meta = emb[:,SEQ_SIZE:]
    meta[:,0] = mass / mass_scale
    for i in range(len(pep)):
        indexed = AA_STRING.index(pep[i])
        emb[i][indexed] = 1.
        emb[i][SEQ_SIZE+1] = mods[i]

    return emb

def embed_all(inp, mass_scale = MAX_MZ):
    ret = numpy.zeros((len(inp), PROT_STR_LEN+1, INP_SIZE), dtype='float32')
    for i in range(len(inp)):
        ret[i] = embed(inp[i]['pep'], numpy.asarray(inp[i]['mods'], dtype='float32'), float(inp[i]['mass']), mass_scale=float(mass_scale))
    return ret

EMB_DIM = 1024

class Res(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(EMB_DIM, EMB_DIM, kernel_size=1, padding='same')
        self.bn = nn.BatchNorm1d(EMB_DIM, momentum=0.01, eps=1e-3, affine=True, track_running_stats=False)
        self.nl = nn.ReLU()

    def forward(self, x):
        out = self.bn(self.conv(x))
        return self.nl(x + out)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.emb_size = EMB_DIM

        self.enc = nn.Conv1d(INP_SIZE+1, self.emb_size, kernel_size=PROT_STR_LEN+1, padding='same')
        self.enc_bn = nn.BatchNorm1d(self.emb_size, momentum=0.01, eps=1e-3, affine=True, track_running_stats=False)
        self.enc_nl = nn.ReLU()

        self.processors = []
        for _ in range(5):
            self.processors.append(Res())

        self.processors = nn.ModuleList(self.processors)

        self.decoders = []
        for _ in range(MAX_CHARGE):
            self.decoders.append(nn.Conv1d(self.emb_size, PROT_TGT_LEN, kernel_size=1, padding='same'))

        self.decoders = nn.ModuleList(self.decoders)
        self.dec_nl = nn.Sigmoid()
        self.dec_pool = nn.AvgPool1d(PROT_STR_LEN)

    def forward(self, inp):
        inp = torch.cat([inp, (torch.arange(inp.shape[1]) / inp.shape[1]).to(inp.device).view(1, -1, 1).repeat(inp.shape[0], 1, 1)], dim=-1)
        inp = inp.transpose(1, 2)
        this_hid = self.enc_nl(self.enc_bn(self.enc(inp)))

        for i, l in enumerate(self.processors):
            this_hid = l(this_hid)

        ret = torch.stack([(self.dec_pool(self.dec_nl(decoder(this_hid)))).view(this_hid.shape[0], -1) for decoder in self.decoders]).transpose(0, 1)

        return ret

    def save(self, fout):
        torch.save(self.state_dict(), open(fout, 'wb'))
