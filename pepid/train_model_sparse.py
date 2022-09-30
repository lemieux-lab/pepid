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

INIT_PATIENCE = 999

raw_data = tables.open_file("data/pt_pf_round.h5", 'r')

class MsDataset(Dataset):
    def __init__(self, data, idxs, max_charge=7, epoch_len=None, theoretical=True, generated=False, shuffle=False):
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
            seq = self.data.root.meta[self.idxs[idx]]['seq'].decode('utf-8')
            mods = numpy.zeros((len(seq),))
            mods[seq == '1'] = pepid_utils.MASS_O
            mods[seq == 'C'] = pepid_utils.MASS_CAM
            seq = seq.replace('1', 'M')
        spec = self.data.root.spectrum[self.idxs[idx]]
        charge = self.data.root.meta[self.idxs[idx]]['charge']
        mass = self.data.root.meta[self.idxs[idx]]['mass']
        
        enc_seq = embed({"pep": seq, "mods": mods, "charge": charge, 'mass': mass})
          
        out_set = []
        out_set.append(torch.FloatTensor(enc_seq))
        out_set.append(torch.LongTensor([charge-1])) # 0-index the charge...
        out_set.append(torch.FloatTensor(spec))
        #out_set[-1] = torch.sqrt(out_set[-1])
        return tuple(out_set)

def import_all(mod):
    imp = __import__(mod, fromlist=["Model", "embed", "PROT_BLIT_LEN", "PROT_STR_LEN"])
    global Model
    global embed
    global PROT_BLIT_LEN
    global PROT_STR_LEN
    global SEQ_SIZE
    embed = getattr(imp, "embed")
    PROT_BLIT_LEN = getattr(imp, "PROT_BLIT_LEN")
    PROT_STR_LEN = getattr(imp, "PROT_STR_LEN")
    SEQ_SIZE = getattr(imp, "SEQ_SIZE")
    Model = getattr(imp, "Model")

def make_run(mod):
    import_all(mod)
    model = Model()

    if CUDA:
        torch.backends.cudnn.enabled = True
        model.cuda()

    print(model)
    #model.load_state_dict(torch.load("ml/{}.pkl".format(mod)))

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
    dataset_test = MsDataset(raw_data, test_idxs, epoch_len=20000)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    dl_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    loss_fn = torch.nn.CosineSimilarity(dim=-1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    patience = INIT_PATIENCE
    best_loss = float('inf')
    last_epoch = 200
    epoch_idx = 0

    prev_sim = 0

    while epoch_idx < last_epoch and patience > 0:
        loss_value = {'train': 0, 'test': 0}
        cos_sim = {'train': 0, 'test': 0}
        sparsity = {'train': 0, 'test': 0}

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
                seq, charge, spec = batch
                seq = seq.cuda()
                spec = spec[:,:PROT_BLIT_LEN].cuda()
                charge = charge.view(-1).cuda()

                spec = spec.view(-1, spec.shape[-1])
                pred = model(seq).view(spec.shape[0], -1)

                l_expr = -loss_fn(pred, spec).mean() - loss_fn(pred * (pred >= 1e-3), spec).mean()

                loss_value[phase] += l_expr.data.cpu().numpy()
                spec_post = spec / torch.norm(spec, p=2, dim=-1, keepdim=True)
                pred_post = pred / torch.norm(pred, p=2, dim=-1, keepdim=True)
                cos_sim[phase] += (spec_post * pred_post).sum(dim=-1).mean().detach().cpu().numpy()

                if this_dl is dl:
                    l_expr.backward(retain_graph=False)
                    optimizer.step()

                sparsity[phase] += (1.0 - (torch.count_nonzero(pred * (pred >= 1e-3), dim=-1).float() / pred.shape[-1]).mean()).detach().cpu().numpy()

                n += 1

                bar.set_description("[{}] {}: {:.3f} -> L:{:.3f} S:{:.3f}".format(epoch_idx, "T{}ing".format(phase[1:]), cos_sim[phase] / max(n, 1), loss_value[phase] / max(n, 1), sparsity[phase] / max(n, 1)))

            loss_value[phase] /= n
            cos_sim[phase] /= n
            sparsity[phase] /= n

        prev_sim = cos_sim['test']

        if loss_value['test'] < best_loss:
            best_loss = loss_value['test']
            model.save("ml/{}.pkl".format(mod))
            patience = INIT_PATIENCE
        else:
            patience -= 1
 
        epoch_idx += 1

if __name__ == '__main__':
    make_run(sys.argv[1])
