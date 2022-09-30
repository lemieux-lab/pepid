import pepid_utils

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

CUDA = True

BATCH_SIZE = 500
MAX_SIZE = 5000000

PROT_BLIT_MZ = 2000

INIT_PATIENCE = 999

raw_data = tables.open_file("data/pt_pf_round.h5", 'r')

from train_model_sparse import MsDataset, import_all

imp = __import__("spectrum_generator_sparse", fromlist=["Model", "embed", "PROT_BLIT_LEN", "PROT_STR_LEN", "MAX_CHARGE"])
Model = getattr(imp, "Model")

import_all("spectrum_generator_sparse")

model = Model()
if CUDA:
    torch.backends.cudnn.enabled = True
    model.cuda()

model.load_state_dict(torch.load("ml/{}.pkl".format("spectrum_generator_sparse")))

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

dataset_test = MsDataset(raw_data, test_idxs, epoch_len=2500)
dl_test = DataLoader(dataset_test, batch_size=100, shuffle=True, num_workers=0, drop_last=True)

import numpy
model.eval()
def show_one():
    numpy.random.seed()

    for batch in dl_test:
        seq, charge, spec = batch
        seq = seq.cuda()
        spec = spec[:,:20000].cuda()
        charge = charge.view(-1).long().cuda()

        pred = model(seq).view(-1, 20000)

        pred = pred * (pred >= 1e-3)

        print("{}\n".format(((pred / torch.norm(pred, p=2, dim=-1, keepdim=True)) * (spec / torch.norm(spec, p=2, dim=-1, keepdim=True))).sum(dim=-1).mean().detach()))

        break

def get_data(n):
    for batch in dl_test:
        seq, charge, spec = batch
        break
    seq = seq[:n].cuda()
    spec = spec[:n,:20000].numpy()
    charge = charge.view(-1)[:n].cuda()

    return seq, charge, spec

def select(pred, charge):
    ret = torch.stack([pred[i,charge[i],:] for i in range(pred.shape[0])])
    return (ret * (ret >= 1e-5)).detach().cpu().numpy()

if __name__ == '__main__':
    show_one()
