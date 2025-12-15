# simulate DNA sequence data

import random as rn
from scipy.stats import norm

def rand_sequence(seqlen):
    return ''.join([ rn.choice('ACGT') for i in range(seqlen) ])

def mutate_sequence(seq, rate):
    '''
    Introduces random point mutations to sequence at rate of:
    rate
    '''
    N = len(seq)
    positions = rn.sample(range(N), int(N * rate))
    for i in positions:
        seq = seq[:i] + rn.choice('ACGT') + seq[(i+1):]
    return seq

def add_indels(seq, rate):
    '''
    Introduces random indels to sequence at a rate of:
    rate
    and of size:
    poission.rvs(mu, 1)
    '''
    N = len(seq)
    positions = rn.sample(range(N), int(N * rate))
    for i in range(len(positions)):
        pos = positions[i]
        size = int(norm.rvs(0, 5, size=1))
        if size > 0 and size+pos <= len(seq):
            seq = seq[:pos+1] + rand_sequence(size) + seq[pos+1:]
        elif size < 0 and abs(size)+pos <= len(seq):
            seq = seq[:pos+1] + seq[pos+1+abs(size):]
    return seq

def random_DNA_data(anc_seq, nseq, nlab, seqlen, bs_rate, ws_rate, indel_rate):
    maxdat = nseq/nlab
    upper, lower = [int(maxdat+maxdat/2), int(maxdat-maxdat/2)]
    data = []
    sp_seq = anc_seq
    for i in range(nlab):
        subset = rn.randint(lower, upper)
        sp_seq = mutate_sequence(sp_seq, bs_rate)
        if indel_rate:
            sp_seq = add_indels(sp_seq, indel_rate)
        for j in range(subset):
            new_seq = mutate_sequence(sp_seq, ws_rate)
            data.append([i,new_seq])
    print("generated "+str(len(data))+" records")
    return data
