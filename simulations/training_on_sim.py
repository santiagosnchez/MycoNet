# testing

import sys
import os
#from sparse import COO
import numpy as np
import pandas as pd
#import seaborn as sbn
from MycoNet.simulate_DNA_data import *
from MycoNet.recode import *
from MycoNet.make_model import *
import tensorflow as tf

def checkpoint(path):
    checkpoint_dir = os.path.dirname(path)   
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path, save_weights_only=True, verbose=1, save_freq='epoch')
    return cp_callback
 
# anc_seq = base sequence
# nseq = number of total sequences, database size
# nlab = number of total labels or sepcies
# bs_rate = between-species divergence rate
# ws_rate = within-species divergence rate
# indel_rate = False
seqlen = int(sys.argv[1])
total_samp = int(sys.argv[2])
species = int(sys.argv[3])
bs_rate = float(sys.argv[4])
ws_rate = float(sys.argv[5])
epochs = int(sys.argv[6])
kmer = 7

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

directory = "_".join(["sim",str(seqlen),str(total_samp),str(species),str(bs_rate),str(ws_rate)])
print(directory)
if not os.path.exists(directory):
    os.makedirs(directory)

# if os.path.exists(directory+"/data_matrix.npy") and os.path.exists(directory+"/labels.npy"):
#    print("found binary data")
#    print("loading X and labels")
#    X = np.load(directory+"/data_matrix.npy")
#    labels = np.load(directory+"/labels.npy")
#    with open(directory+"/vocab_size.txt", "r") as f:
#        vocab_size = int(f.read().rstrip())
#    print("dimensions; X:",X.shape,", labels:",labels.shape)
#else:

if os.path.exists(directory+'/X_train.npy'):
    X_train = np.load(directory+"/"+"X_train.npy")
    X_test = np.load(directory+"/"+"X_test.npy")
    y_train = np.load(directory+"/"+"y_train.npy")
    y_test = np.load(directory+"/"+"y_test.npy")
    with open(directory+"/vocab_size.txt", "r") as f:
        vocab_size = int(f.read().rstrip())
    # make model
    model = make_model(input_dim=vocab_size+1, output_layer=y_train.shape[0], dropout=0.2, embedding_dim=32, LSTM_dim=32, learning_rate=0.001, activation=0.01)
    # load checkpoint
    print("loading weights ...")
    model.load_weights(directory+'/'+directory+".cp.ckpt")
    cp_callback = checkpoint(directory+'/'+directory+".cp.ckpt")
    # run model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=1000, validation_data=(X_test, y_test), verbose = 1, callbacks=[cp_callback])
    model.save(directory+'/'+directory+"_entire_model") 
    # save history
    pd.DataFrame.from_dict(history.history).to_csv(directory+'/history.csv', mode='a', header=False, index=False)
else:
    print("generating data")
    data = random_DNA_data(rand_sequence(seqlen), nseq=total_samp, nlab=species, seqlen=seqlen, bs_rate=bs_rate, ws_rate=ws_rate, indel_rate=None)
    # make it sparse
    #labels = tf.one_hot([h for h,s in data], 100, dtype=np.int8)
    #labels = get_labels(data, flatten=True)
    labels = np.array([ h for h,s in data ], np.int32) 
    X,vocab_size = get_recoded_sequences(data, pad=False, longest=None, kmer=kmer)
    del data
    #print("saving binary data")
    #np.save(directory+"/data_matrix.npy", X)
    #np.save(directory+"/labels.npy", labels)
    with open(directory+"/vocab_size.txt", "w") as o:
        o.write(str(vocab_size)+"\n")
    print("dimensions; X:",X.shape,", labels:",labels.shape)
    #X = get_recoded_sequences(data, pad=True, longest=max(seq_lengths))
    X_train,X_test,y_train,y_test = split_train_test(X, labels, test_size=0.2)
    np.save(directory+"/"+"X_train.npy", X_train)
    np.save(directory+"/"+"X_test.npy", X_test)
    np.save(directory+"/"+"y_train.npy", y_train)
    np.save(directory+"/"+"y_test.npy", y_test)
    # garbage collection
    del X
    del labels
    # create checkpoints
    cp_callback = checkpoint(directory+'/'+directory+".cp.ckpt")
    model = make_model(input_dim=vocab_size+1, output_layer=y_train.shape[0], dropout=0.2, embedding_dim=32, LSTM_dim=32, learning_rate=0.001, activation=0.01)   
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=1000, validation_data=(X_test, y_test), verbose = 1, callbacks=[cp_callback])
    model.save(directory+'/'+directory+"_entire_model") 
    # save history
    pd.DataFrame.from_dict(history.history).to_csv(directory+'/history.csv', mode='a', header=False, index=False)

