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

INP_SIZE = SEQ_SIZE+1+2

def embed(inp, mass_scale = 2000):
# Input is {"pep": peptide, "charge": charge, "mass": precursor mass, "type": 3 = 'hcd', "nce": 25"}
#n_meta = 8
    emb = numpy.zeros((PROT_STR_LEN + 1, INP_SIZE), dtype='float32')

    mass = pepid_utils.neutral_mass(inp['pep'], inp['mods'], NTERM, CTERM, inp['charge'])

    pep = inp['pep']

    for i in range(len(pep), PROT_STR_LEN): emb[i,pepid_utils.AMINOS.index("_")] = 1. # padding first, meta column should not be affected
    meta = emb[:,SEQ_SIZE+1:]
    meta[:,0] = mass / mass_scale #pepid_utils.neutral_mass(inp['pep'], inp['mods'], NTERM, CTERM, z=1) / mass_scale
    meta[:,1] = inp['charge']
    for i in range(len(pep)):
        emb[i,pepid_utils.AMINOS.index(pep[i])] = 1.
        emb[i,SEQ_SIZE] = inp['mods'][i]

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

        self.emb_size = 100

        self.enc_query1 = nn.Linear(PROT_BLIT_LEN, self.emb_size*4)
        self.enc_query2 = nn.Linear(self.emb_size*4, self.emb_size*2)
        self.enc_query3 = nn.Linear(self.emb_size*2, self.emb_size)
        self.enc_gt1 = nn.Linear(PROT_BLIT_LEN, self.emb_size*4)
        self.enc_gt2 = nn.Linear(self.emb_size*4, self.emb_size*2)
        self.enc_gt3 = nn.Linear(self.emb_size*2, self.emb_size)
        self.score1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.score2 = nn.Linear(self.emb_size, self.emb_size//2)
        self.score3 = nn.Linear(self.emb_size//2, 1)
        self.out_nl = nn.Sigmoid()
        self.nl = nn.ReLU()

    def forward(self, query, gt):
        query = self.nl((self.enc_query1(query)))
        query = self.nl((self.enc_query2(query)))
        query = self.nl((self.enc_query3(query)))
        gt = self.nl((self.enc_gt1(gt)))
        gt = self.nl((self.enc_gt2(gt)))
        gt = self.nl((self.enc_gt3(gt)))
        score = self.nl((self.score1(torch.cat((query, gt), dim=1))))
        score = self.nl((self.score2(score)))
        score = self.out_nl(self.score3(score))
        return score

    def save(self, fout):
        torch.save(self.state_dict(), open(fout, 'wb'))
