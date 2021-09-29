import random
import os
import torch
from sklearn.cluster import KMeans
from transformers import BertModel, BertTokenizer
import torchtext
import numpy as np 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import pandas as pd

slot_list = ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist','city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 
    'poi', 'sort', 'spatial_relation', 'state', 'party_size_description','city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description', 'genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist',
'object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating',
'object_name', 'object_type', 'timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
def get_random():
    slot_list1 = list(set(slot_list))
    mp = {1:'A',2:'B',3:'C',4:'D',5:'E',6:'F'}
    res_dict = {'pad':['<PAD>'],'O':['O'],'A':[], 'B':[], 'C':[],'D':[],'E':[],'F':[]}
    for i, slot in enumerate(slot_list1):
        t = random.randint(1,6)
        res_dict[mp[t]].append(slot)
    return res_dict

datadir = '/home/shenhao/data/coachdata/snips/'
glovepath = '/home/shenhao/data'
charNgrampath = '/home/shenhao/data'
bertpath = '/home/shenhao/bert-base-uncased'

domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather",\
     "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

sz=50

def get_data(datadir):
    data = []
    slot2values = {}
    for domain in domain_set:
        p = os.path.join(datadir, domain, domain+'.txt')
        for line in open(p, 'r'):
            line = line.strip()
            toks, labels = line.split('\t')
            toks = toks.split()
            labels = labels.split()
            assert len(toks) == len(labels)
            data.append((toks, labels))
    
    for toks, labels in data:
        i = 0
        while i < len(labels):
            if labels[i] != 'O':
                j = i+1
                while j < len(labels) and labels[j][0] == 'I' \
                and labels[j][2:] == labels[i][2:]:
                    j += 1
                
                if labels[i][2:] not in slot2values:
                    slot2values[labels[i][2:]] = []

                slot2values[labels[i][2:]].append(toks[i:j])
                i = j
            else: i += 1

    for k in slot2values.keys():
        random.shuffle(slot2values[k])
        slot2values[k] = slot2values[k][:sz]
    return slot2values

def slot2emb(data):
    Bertemb = {}
    Gloveemb = {}
    charNgramemb = {}
    tokenizer = BertTokenizer.from_pretrained(bertpath)
    encoder = BertModel.from_pretrained(bertpath)
    encoder = encoder.to('cuda:1')
    
    
    for k,v in data.items():
        for temp in v:
            toks = tokenizer.encode(' '.join(temp))
            toks = torch.tensor(toks, device='cuda:1')
            toks = toks.unsqueeze(0)
            reps = encoder(toks)[0].mean(1).squeeze()
            reps = reps.detach().cpu().numpy()
            if k not in Bertemb:
                Bertemb[k] = []
            Bertemb[k].append(np.reshape(reps, (768, 1)))
        Bertemb[k] = np.concatenate(Bertemb[k], -1)
    
    glove = torchtext.vocab.GloVe(cache=glovepath, name='6B')
    for k,v in data.items():
        for temp in v:
            slot_tokens = []
            for i in temp:
                slot_tokens.append(np.reshape(glove[i], (300,1)))
            slot_tokens = np.concatenate(slot_tokens, -1)
            reps = np.mean(slot_tokens,-1)
            if k not in Gloveemb:
                Gloveemb[k] = []
            Gloveemb[k].append(np.reshape(reps, (300, 1)))
        Gloveemb[k] = np.concatenate(Gloveemb[k], -1)
            
    char_ngram_model = torchtext.vocab.CharNGram(cache=charNgrampath)
    for k,v in data.items():
        for temp in v:
            slot_tokens = []
            for i in temp:
                slot_tokens.append(np.reshape(char_ngram_model[i], (100,1)))
            slot_tokens = np.concatenate(slot_tokens, -1)
            reps = np.mean(slot_tokens,-1)
            if k not in charNgramemb:
                charNgramemb[k] = []
            charNgramemb[k].append(np.reshape(reps, (100, 1)))
        charNgramemb[k] = np.concatenate(charNgramemb[k], -1)
    
    
    return Bertemb, Gloveemb, charNgramemb

def Kmeans(emb):
    temp_embs = []
    for k,v in emb.items():
        temp_embs.append(v)
    temp_embs = np.concatenate(temp_embs, -1)
    temp_embs = temp_embs.transpose(1,0)
    

    kmeans = KMeans(n_clusters=6, random_state= 0).fit(temp_embs)
    y_ = list(kmeans.labels_)
    # print(y_)
    slot_name = []
    coarse_label = []
    t = 0
    temp_y = []
    for i in range(len(y_)):
        slot_name.append(t)
        temp_y.append(y_[i])
        if (i+1) % sz == 0:
            t += 1
            coarse_label.append(max(temp_y, key=temp_y.count))
            temp_y.clear()
    coarse_label_dict = {}
    t = 0
    for k, v in emb.items():
        coarse_label_dict[k] = coarse_label[t]
        t += 1

    mp = {0:'A', 1:'B', 2:'C',3:'D',4:'E',5:'F'}
    for k in coarse_label_dict:
        coarse_label_dict[k] = mp[coarse_label_dict[k]]
    
    coarse2slot = {'pad':['<PAD>'], 'O':['O']}
    for k, v in coarse_label_dict.items():
        if v not in coarse2slot:
            coarse2slot[v] = []
        coarse2slot[v].append(k)

    return slot_name, y_, coarse_label, coarse_label_dict, coarse2slot

def get_cluster():
    data = get_data(datadir)
    bert_reps, glove_reps, charN_reps = slot2emb(data)
    slot_name, y, coarse_label, coarse_label_dict, coarse2slot = Kmeans(bert_reps)
    return coarse2slot

