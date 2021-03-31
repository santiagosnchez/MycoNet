# testing

import os
import sys
import numpy as np
import pandas as pd
#import seaborn as sbn
#from simulate_DNA_data import *
from MycoNet.recode import *
from MycoNet.kmer_embedding import *
from MycoNet.make_model import *
#from sklearn.model_selection import train_test_split
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.utils import to_categorical
import tensorflow as tf

fasta = sys.argv[1]
model_dir = sys.argv[2]
#unite_sh = sys.argv[3]
outfile = sys.argv[3]
model_path = model_dir + "/training/" + model_dir + "_entire_model"
labels_key_dict = model_dir + "/inputs/labels_key_dict.txt"
kmer_dict = model_dir + "/inputs/kmer_dict.txt"
maxseq = 1500 
minseq = 100
kmer = 10

def split_data(fasta):
    with open(fasta) as f:
        data = f.read().splitlines()
    idx = iter(range(len(data)))
    d = [ [data[i],data[j].upper()] for i,j in zip(idx,idx) ]
    return d

def preprocessdata(fasta, maxlen, minlen):
    # read data and filter
    print("splitting data")
    data = split_data(fasta)
    # seq lengths
    seqlens = np.array([ len(s) for h,s in data ])
    print("max sequence length:", seqlens.max())
    print("min sequence length:", seqlens.min())
    print("filtering")
    print("capping to: max <=",maxlen,", min >=",minlen)
    data = [ [x,y] for x,y in data if len(y) <= maxlen and len(y) >= minlen ]
    return data

def predict_batch(model, X, batch, labels, outfile, df_labels_key_dict): #, df_unite_sh):
    batch_list = range(0, X.shape[0], batch)
    for b in batch_list:
        if b+batch < X.shape[0]:
            Xb = X[b:(b+batch)]
            lb = labels[b:(b+batch)]
        else:
            Xb = X[b:]
            lb = labels[b:]
        pred = model.predict(Xb)
        best = [ x.argmax() for x in pred ]
        best_prob = [ x.max() for x in pred ]
        best_labs = df_labels_key_dict['header'][best].tolist()
        #best_tax = df_unite_sh['taxonomy'][best_labs].tolist()
        df = pd.DataFrame(list(zip(best_labs, best_prob, lb)), columns=["SH","prob","org_label"])#"taxonomy"])
        df.to_csv(outfile, mode='a', header=True, index=False) 
        print("batch:",b)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

data = preprocessdata(fasta, maxseq, minseq)
#labels_flat = preprocesslabels(data)

#x = df_kmer_dict.to_dict()["int"]
df_labels_key_dict = pd.read_csv(labels_key_dict, sep='\t', header=None, names=("int","header"), index_col=0)
df_kmer_dict = pd.read_csv(kmer_dict, sep='\t', header=None, names=("kmer","int"), index_col=0)
#df_unite_sh = pd.read_csv(unite_sh, sep='\t', header=None, names=("SH","taxonomy"), index_col=0)

# process recoded data
print("recoding DNA and padding")
X,missing = kmer_to_int([ s for h,s in data], k=kmer, kmer_key=df_kmer_dict.to_dict()["int"], padding=True, max_size=maxseq)
print("missing kmers:",len(missing))
labels = [ h for h,s in data ]
print("final X dim:", X.shape)

#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():
model = tf.keras.models.load_model(model_path)

#df_labels_key_dict.iloc[0:10].to_string(index=False, header=False)

if os.path.exists(outfile):
    print(outfile, "found. Deleting...")
    os.remove(outfile)
predict_batch(model=model, X=X, batch=1000, labels=labels, outfile=outfile, df_labels_key_dict=df_labels_key_dict) #, df_unite_sh=df_unite_sh)


