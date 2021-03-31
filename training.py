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

fasta_db = sys.argv[1]
name = sys.argv[2]
kmer = int(sys.argv[3])
maxseq = 1500 
minseq = 100
test_size = 0.2; dropout=0.2; embedding_dim=32; activation=0.01; learning_rate=0.001; LSTM_dim=32
epochs=500; batch_size=1000


def checkpoint(path):
    checkpoint_dir = os.path.dirname(path)   
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path, save_weights_only=True, verbose=1, save_freq='epoch')
    return cp_callback

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

def preprocesslabels(data, name):
    print("processing labels")
    labels_short = [ x[0].split("|")[-1] for x in data ]
    labels_unique = list(set(labels_short))
    labels_dict = { y:x for x,y in enumerate(labels_unique) }
    labels_dict_back = { x:y for x,y in enumerate(labels_unique) }
    with open(name+"/inputs/labels_key_dict.txt", "w") as o:
        for i in labels_dict_back.keys():
            o.write(str(i) + "\t" + labels_dict_back[i] + "\n") 
    labels_numb = [ labels_dict[x] for x in labels_short ]
    #labels_flat = flatten_labels(labels_numb)
    return labels_numb

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


if not os.path.exists(name):
    os.makedirs(name)

if not os.path.exists(name+"/inputs"):
    os.makedirs(name+"/inputs")

if os.path.exists(name+"/inputs/X_train.npy"):
    print("loading testing and training sets")
    X_train = np.load(name+"/inputs/X_train.npy")
    y_train = np.load(name+"/inputs/y_train.npy")
    X_test = np.load(name+"/inputs/X_test.npy")
    y_test = np.load(name+"/inputs/y_test.npy")
    with open(name+"/inputs/vocab_size.txt","r") as f:
        vocab_size = int(f.read().rstrip())
elif os.path.exists(name+"/inputs/data_sequence_matrix.npy") and os.path.exists(name+"inputs/labels_matrix.npy"):
    print("loading data matrix")
    X = np.load(name+"/inputs/data_sequence_matrix.npy")
    print("data dimensions:", X.shape)
    print("loading labels matrix")
    labels_idx = np.load(name+"/inputs/labels_matrix.npy")
    print("labels dimensions:", labels_idx.shape)
    with open(name+"/inputs/vocab_size.txt","r") as f:
        vocab_size = int(f.read().rstrip())
else:
    # process data
    data = preprocessdata(fasta_db, maxseq, minseq)
    labels_idx = preprocesslabels(data, name)
    np.save(name+"/inputs/labels_matrix.npy", labels_idx)
    
    # process recoded data
    print("recoding DNA and padding")
    X,vocab_size = get_recoded_sequences(data, pad=True, longest=maxseq, kmer=kmer)
    print("final X dim:", X.shape)
    print("final labels dim:", labels_flat.shape)
    np.save(name+"/inputs/data_sequence_matrix.npy", X)
    with open(name+"/inputs/vocab_size.txt","w") as o:
        o.write(str(vocab_size) + "\n")

# split testing/training
try:
    len(X_train) == 0
except NameError:  
    print("splitting into testing and training sets, test frac:",test_size)
    X_train,X_test,y_train,y_test = split_train_test(X, labels_idx, test_size=test_size)
    np.save(name+"/inputs/X_train.npy", X_train)
    np.save(name+"/inputs/X_test.npy", X_test)
    np.save(name+"/inputs/y_train.npy", y_train)
    np.save(name+"/inputs/y_test.npy", y_test)

# data sizes
print("data sizes:")
print("X_train: ", X_train.nbytes/1e6, "MB", X_train.shape)
print("X_test:", X_test.nbytes/1e6, "MB,", X_test.shape)
print("y_train:", y_train.nbytes/1e6, "MB", y_train.shape)
print("y_test:", y_test.nbytes/1e6, "MB", y_test.shape)
print("total training size:", (X_train.nbytes + y_train.nbytes)/1e6, "MB")
print("all data:", X_train.shape + X_test.shape, y_train.shape + y_test.shape)


# run
#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():
model = make_model(input_dim=vocab_size+1, output_layer=y_train.shape[0], dropout=dropout, embedding_dim=embedding_dim, activation=activation, learning_rate=learning_rate, LSTM_dim=LSTM_dim)
if os.path.exists(name+"/training/"+name+"_entire_model"):
    model = tf.keras.models.load_model(name+"/training/"+name+".cp.ckpt")
elif os.path.exists(name+"/training/"+name+".cp.ckpt.index"):
    print("checkpoint found, loading weights")
    model.load_weights(name+"/training/"+name+".cp.ckpt")

print("making checkpoint")
cp_callback = checkpoint(name+"/training/"+name+".cp.ckpt")
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose = 1, callbacks=[cp_callback])

# Save the entire model as a SavedModel.
model.save(name+'/training/'+name+"_entire_model") 

# save history
pd.DataFrame.from_dict(history.history).to_csv(name+'/training/history.csv', mode='a', header=False, index=False)


