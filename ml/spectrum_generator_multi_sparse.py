import pepid_utils

import configparser
config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read('data/default.cfg')
config.read('example.cfg')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import numpy
import tables
import pickle as cPickle

import tqdm

import os
import sys

NTERM, CTERM = config['processing.db'].getfloat('nterm cleavage'), config['processing.db'].getfloat('cterm cleavage')

SIZE_RESOLUTION_FACTOR = 10
PROT_BLIT_LEN = 2000 * SIZE_RESOLUTION_FACTOR

AA_LIST = pepid_utils.AMINOS
SEQ_SIZE = len(AA_LIST)

PROT_STR_LEN = config['processing.db'].getint('max length')
MAX_CHARGE = config['processing.db'].getint('max charge')
N_MASS_REPS = MAX_CHARGE

INP_SIZE = SEQ_SIZE+2

def embed(inp, mass_scale = 2000):
# Input is {"pep": peptide, "charge": charge, "mass": precursor mass, "type": 3 = 'hcd', "nce": 25"}
#n_meta = 8
    emb = numpy.zeros((PROT_STR_LEN + 1, INP_SIZE), dtype='float32')

    pep = inp['pep']

    for i in range(len(pep), PROT_STR_LEN): emb[i,pepid_utils.AMINOS.index("_")] = 1. # padding first, meta column should not be affected
    meta = emb[:,SEQ_SIZE:]
    meta[:,0] = pepid_utils.neutral_mass(inp['pep'], inp['mods'], NTERM, CTERM, z=1) / mass_scale
    for i in range(len(pep)):
        emb[i,pepid_utils.AMINOS.index(pep[i])] = 1.
        emb[i,SEQ_SIZE+1] = inp['mods'][i]

    return emb

#def embed(inp):
#    # Input is [{"pep": peptide, "mods": mods array, "charge": charge, "mass": precursor mass}]
#    embedding = numpy.zeros((PROT_STR_LEN + 1, INP_SIZE), dtype='float32')
#    embedding[len(inp['pep']):PROT_STR_LEN,pepid_utils.AMINOS.index("_")] = 1
#    embedding[:,SEQ_SIZE:SEQ_SIZE+1] = 0 # default mods = 0
#    embedding[:,-N_MASS_REPS] = pepid_utils.neutral_mass(inp['pep'], inp['mods'], NTERM, CTERM, z=1)
#    for i in range(len(inp['pep'])):
#        embedding[i,pepid_utils.AMINOS.index(inp['pep'][i])] = 1.
#    embedding[:len(inp['mods']),SEQ_SIZE] = inp['mods']
#
#    return embedding

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.emb_size = 1024

        self.enc = nn.Conv1d(INP_SIZE, self.emb_size, kernel_size=PROT_STR_LEN // 2 * 2 + 1, padding=PROT_STR_LEN // 2)
        self.enc_bn = nn.BatchNorm1d(self.emb_size, momentum=0.01, eps=1e-3, affine=True, track_running_stats=False)
        self.enc_nl = nn.ReLU()

        self.processors = []
        for _ in range(5):
            self.processors.append(nn.Conv1d(self.emb_size, self.emb_size, kernel_size=3, padding=1))
            self.processors.append(nn.BatchNorm1d(self.emb_size, momentum=0.01, eps=1e-3, affine=True, track_running_stats=False))
            self.processors.append(nn.ReLU())
        self.processors = nn.ModuleList(self.processors)

        self.decoders = []
        for _ in range(MAX_CHARGE):
            self.decoders.append(nn.Conv1d(self.emb_size, PROT_BLIT_LEN, kernel_size=1, padding=0))
        self.decoders = nn.ModuleList(self.decoders)
        self.dec_nl = nn.Sigmoid()
        self.dec_pool = nn.AvgPool1d(PROT_STR_LEN)

    def forward(self, inp):
        inp = inp.transpose(1, 2)
        this_hid = self.enc_nl(self.enc_bn(self.enc(inp)))

        for i, l in enumerate(self.processors):
            this_hid = l(this_hid)

        ret = torch.stack([(self.dec_pool(self.dec_nl(decoder(this_hid)))).transpose(2, 1).view(this_hid.shape[0], -1) for decoder in self.decoders]).transpose(0, 1)

        return ret

    def save(self, fout):
        torch.save(self.state_dict(), open(fout, 'wb'))
