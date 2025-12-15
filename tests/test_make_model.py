import numpy as np
from MycoNet import make_model as mm


def test_make_model_structure_and_output_shape():
    # small params to keep model tiny
    vocab = 50
    emb_dim = 8
    out_classes = 3
    dropout = 0.1
    lr = 0.01
    lstm_dim = 4

    model = mm.make_model(vocab, emb_dim, out_classes, dropout, "relu", lr, lstm_dim)
    # check layers by name
    names = [layer.name for layer in model.layers]
    assert "embedding_layer" in names
    assert "output_layer" in names

    # create dummy input batch: batch_size x seq_len
    X = np.random.randint(1, vocab, size=(2, 10))
    preds = model.predict(X)
    assert preds.shape == (2, out_classes)


def test_split_train_test():
    X = np.arange(20).reshape(10, 2)
    labels = np.arange(10)
    Xtr, Xte, ytr, yte = mm.split_train_test(X, labels, test_size=0.3)
    assert Xtr.shape[0] + Xte.shape[0] == X.shape[0]
    assert ytr.shape[0] + yte.shape[0] == labels.shape[0]
