from itertools import product
import numpy as np
import re
import random as rn
import collections as cl

def all_kmers_dict(k):
    # https://stackoverflow.com/a/48677719/1706987
    all_possible = [''.join(c) for c in product('ACTG', repeat=k)]
    kmer_key = { y:x for x,y in list(enumerate(all_possible, start=1))}
    return kmer_key

def find_all_kmers(seqs, kmer):
    def build_kmer_dict(kmer_dict, seq, kmer):
        for i in range(len(seq)-kmer):
            kmer_dict[seq[i:(i+kmer)]] = 0
        return kmer_dict
    kmer_dict = cl.defaultdict()
    for seq in seqs: kmer_dict = build_kmer_dict(kmer_dict, resolve_amb(seq), kmer)
    i = 1
    #with open("kmer_dict.txt", "w") as o:
    #    for k in kmer_dict.keys():
    #        kmer_dict[k] = i
    #        o.write(k+"\t"+str(i)+"\n")
    #        i += 1
    return kmer_dict

#def resolve_amb(seq):
#    return re.sub('[^ATGC]+', rn.choice('ACTG'), seq)

#def resolve_iupac():
#    return 'AGTC' + rn.choice('AC') + \
#            rn.choice('AG') + rn.choice('AT') + \
#            rn.choice('CG') + rn.choice('CT') + \
#            rn.choice('GT') + rn.choice('ACG') + \
#            rn.choice('ACT') + rn.choice('AGT') + \
#            rn.choice('CGT') + rn.choice('AGTC') 

#def resolve_amb2(seq):
#   trans = seq.maketrans('ACGTMRWSYKVHDBN',resolve_iupac())
#   return seq.translate(trans) 

def resolve_amb(seq):
    if 'M' in seq:
        seq = re.sub('M',rn.choice('AC'), seq)
    if 'R' in seq:
        seq = re.sub('R',rn.choice('AG'), seq)
    if 'W' in seq:
        seq = re.sub('W',rn.choice('AT'), seq)
    if 'S' in seq:
        seq = re.sub('S',rn.choice('CG'), seq)
    if 'Y' in seq:
        seq = re.sub('Y',rn.choice('CT'), seq)
    if 'K' in seq:
        seq = re.sub('K',rn.choice('GT'), seq)
    if 'V' in seq:
        seq = re.sub('V',rn.choice('ACG'), seq)
    if 'H' in seq:
        seq = re.sub('H',rn.choice('ACT'), seq)
    if 'D' in seq:
        seq = re.sub('D',rn.choice('AGT'), seq)
    if 'B' in seq:
        seq = re.sub('B',rn.choice('CGT'), seq)
    if 'N' in seq:
        seq = re.sub('N',rn.choice('ACGT'), seq)
    return seq

def kmer_to_int(seqs, k, kmer_key, padding, max_size):
    m = []
    missing_kmers = []
    if padding:
        for seq in seqs:
            trans = []
            for i in range(len(seq)-k):
                mer = seq[i:(i+k)]
                try:
                    trans.append(kmer_key[mer])
                except KeyError:
                    trans.append(0)
                    missing_kmers.append(mer)
            zero_pad = [0] * ((max_size-k) - len(trans))
            m.append(zero_pad + trans)
        if len(kmer_key) < 32767:
            return np.array(m, np.int16), missing_kmers
        else:
            return np.array(m, np.int32), missing_kmers
    else:
        for seq in seqs:
            trans = []
            for i in range(len(seq)-k):
                mer = seq[i:(i+k)]
                try:
                    trans.append(kmer_key[mer])
                except KeyError:
                    trans.append(0)
                    missing_kmers.append(mer)
            m.append(trans)
        if len(kmer_key) < 32767:
            return np.array(m, np.int16),missing_kmers
        else:
            return np.array(m, np.int32),missing_kmers


