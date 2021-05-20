import os
import subprocess
import pickle
import logging
import time
import random
import torchtext
from datetime import timedelta

import numpy as np
from tqdm import tqdm
import pickle
char_ngram_model = torchtext.vocab.CharNGram(cache = '/home/shenhao/data')

def load_embedding(vocab, emb_dim, emb_file, oov_emb_file=""):
    # logger = logging.getLogger()
    embedding = np.zeros((vocab.n_words, emb_dim))
    print("embedding: %d x %d" % (vocab.n_words, emb_dim))
    assert emb_file is not None
    with open(emb_file, "r") as ef:
        print('Loading embedding file: %s' % emb_file)
        pre_trained = 0
        embedded_words = []
        for i, line in enumerate(ef):
            if i == 0: continue # first line would be "num of words and dimention"
            line = line.strip()
            sp = line.split()
            try:
                assert len(sp) == emb_dim + 1
            except:
                continue
            if sp[0] in vocab.word2index and sp[0] not in embedded_words:
                pre_trained += 1
                embedding[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]] 
                embedded_words.append(sp[0])
        
        if oov_emb_file == "":
            print("Pre-train: %d / %d (%.2f)" % (pre_trained, vocab.n_words, pre_trained / vocab.n_words))

    if oov_emb_file != "":
        with open(oov_emb_file, "r") as oef:
            print('Loading OoV embedding file: %s' % emb_file)
            for i, line in enumerate(oef):
                line = line.strip()
                sp = line.split()
                if sp[0] in vocab.word2index and sp[0] not in embedded_words:
                    pre_trained += 1
                    embedding[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]] 
                    embedded_words.append(sp[0])
            
            print("Pre-train: %d / %d (%.2f)" % (pre_trained, vocab.n_words, pre_trained / vocab.n_words))

    return embedding

def load_embedding_from_npy(emb_file):
    logger = logging.getLogger()
    logger.info('Loading embedding file: %s' % emb_file)

    embedding = np.load(emb_file)

    return embedding

def load_embedding_from_pkl(emb_file):
    logger = logging.getLogger()
    logger.info('Loading embedding file: %s' % emb_file)

    with open(emb_file, "rb") as f:
        embedding = pickle.load(f)

    return embedding