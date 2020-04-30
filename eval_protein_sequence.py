import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import torch.utils.data

from src.alphabets import Uniprot21

def parse_2line(f, alphabet):
    names = []
    seqs = []
    for line in f:
        if line.startswith(b'>'):
            name = line[1:].decode('utf-8').rstrip()
            names.append(name)

            seq = f.readline().strip()
            if name.startswith('CDK4-'):
                seq2 = f.readline().strip()
                seqs.append((alphabet.encode(seq), alphabet.encode(seq2)))
            else:
                seqs.append(alphabet.encode(seq))
    return names, seqs

def load_2line(path, alphabet):
    with open(path, 'rb') as f:
        names, seqs = parse_2line(f, alphabet)
    return names, seqs

def load_data():
    alphabet = Uniprot21()

    path = 'data/sarkisyan2016gfp/fpbase.fa'#'data/davis2011kinase/uniprot_sequences_processed.fasta'
    names, seqs = load_2line(path, alphabet)

    datasets = {'dataset0': (seqs, names)}

    return datasets

def split_dataset(xs, ys, random=np.random, k=5):
    x_splits = [[] for _ in range(k)]
    y_splits = [[] for _ in range(k)]
    order = random.permutation(len(xs))
    for i in range(len(order)):
        j = order[i]
        x_s = x_splits[i%k]
        y_s = y_splits[i%k]
        x_s.append(xs[j])
        y_s.append(ys[j])
    return x_splits, y_splits

def unstack_lstm(lstm):
    in_size = lstm.input_size
    hidden_dim = lstm.hidden_size
    layers = []
    for i in range(lstm.num_layers):
        layer = nn.LSTM(in_size, hidden_dim, batch_first=True, bidirectional=True)
        attributes = ['weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l']
        for attr in attributes:
            dest = attr + '0'
            src = attr + str(i)
            getattr(layer, dest).data[:] = getattr(lstm, src)
            #setattr(layer, dest, getattr(lstm, src))

            dest = attr + '0_reverse'
            src = attr + str(i) + '_reverse'
            getattr(layer, dest).data[:] = getattr(lstm, src)
            #setattr(layer, dest, getattr(lstm, src))
        layers.append(layer)
        in_size = 2*hidden_dim
    return layers

def featurize(x, lm_embed, lstm_stack, proj, include_lm=True, lm_only=False):
    zs = []

    x_onehot = x.new(x.size(0),x.size(1), 21).float().zero_()
    x_onehot.scatter_(2,x.unsqueeze(2),1)
    zs.append(x_onehot)

    h = lm_embed(x)
    if include_lm:
        zs.append(h)
    if not lm_only:
        for lstm in lstm_stack:
            h,_ = lstm(h)
            zs.append(h)
        h = proj(h.squeeze(0)).unsqueeze(0)
        zs.append(h)
    z = torch.cat(zs, 2)
    return z

def featurize_dict(datasets, lm_embed, lstm_stack, proj,
                   use_cuda=False, include_lm=True, lm_only=False):
    z = {}
    for k,v in datasets.items():
        x_k = v[0]
        names = v[1]
        z[k] = []
        with torch.no_grad():
            for x, name in zip(x_k, names):
                if name.startswith('CDK4-'):
                    x, x2 = x
                x = torch.from_numpy(x).long().unsqueeze(0)
                if use_cuda:
                    x = x.cuda()
                z_x = featurize(x, lm_embed, lstm_stack, proj,
                                include_lm=include_lm, lm_only=lm_only)
                z_x = z_x.squeeze(0).cpu()
                if name.startswith('CDK4-'):
                    x2 = torch.from_numpy(x2).long().unsqueeze(0)
                    if use_cuda:
                        x2 = x2.cuda()
                    z_x2 = featurize(x2, lm_embed, lstm_stack, proj,
                                     include_lm=include_lm, lm_only=lm_only)
                    z_x2 = z_x2.squeeze(0).cpu()
                    z_x = torch.cat((z_x, z_x2), 0)
                z[k].append(z_x)
    return z


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to saved embedding model')
    parser.add_argument('--hidden-dim', type=int, default=150, help='dimension of LSTM (default: 150)')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of training epochs (default: 10)')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    args = parser.parse_args()

    datasets = load_data()
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim

    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)


    ## load the embedding model
    encoder = torch.load(args.model)
    encoder.eval()
    encoder = encoder.embedding

    lm_embed = encoder.embed
    lstm_stack = unstack_lstm(encoder.rnn)
    proj = encoder.proj

    if use_cuda:
        lm_embed.cuda()
        for lstm in lstm_stack:
            lstm.cuda()
        proj.cuda()

    ## featurize the sequences
    z = featurize_dict(datasets, lm_embed, lstm_stack, proj, use_cuda=use_cuda)

    embeddings = z['dataset0']
    names = datasets['dataset0'][1]
    assert(len(embeddings) == len(names))
    for name, embedding in zip(names, embeddings):
        print('>{}'.format(name))
        print('\t'.join([ str(val) for val in np.array(embedding.mean(0)) ]))

if __name__ == '__main__':
    main()
