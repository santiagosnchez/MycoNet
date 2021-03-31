import numpy as np
# from keras.preprocessing.text import Tokenizer
# from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from MycoNet.kmer_embedding import *
from scipy.sparse import lil_matrix

def flatten_DNA(seq):
    trans = seq.maketrans('ACGTMRWSYKVHDBN','123411122300000')
    flat = [ int(i) for i in list(seq.translate(trans)) ]
    return flat

def flatten_labels(labels):
    lab = np.zeros((len(labels), max(labels)+1), np.int8)
    #lab = lil_matrix((len(labels), max(labels)+1), dtype=np.int8)
    for i in range(len(labels)):
        lab[i,labels[i]] = 1
    return lab

def get_labels(data, flatten):
    if flatten:
        return flatten_labels([ h for h,s in data ])
    else:
        return [ h for h,s in data ]

def get_recoded_sequences(data, pad, longest, kmer):
    if kmer:
        print("using a kmer approach, kmer size:",kmer,"bp")
        seqs = [ resolve_amb(s) for h,s in data ]
        print("searching for kmers in sequence data")
        exp_kmer_key = all_kmers_dict(kmer)
        kmer_key = find_all_kmers(seqs, kmer)
        X,missing = kmer_to_int(seqs, kmer, exp_kmer_key, pad, longest)
        print('missing kmers:',len(missing))
        print('expected kmer dict:', len(exp_kmer_key))
        print('observed kmer dict:', len(kmer_key))
        return X,len(exp_kmer_key)
    else:
        X = np.array([ np.array(flatten_DNA(s)) for h,s in data ])
        if pad:
            X = pad_sequences(X, maxlen = longest)
        return X.astype(np.int8),4
