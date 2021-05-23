from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
slot_list = ['<PAD>','price', 'leave',
 'food', 'time', 'day', 'type', 'arrive', 'people', 
 'depart', 'stars', 'stay', 'name', 'dest', 'area']

y_set = ['<PAD>' ,'O', 'B-price', 'I-price',
    'B-leave','I-leave', 
    'B-food', 'I-food',  
    'B-time','I-time', 
    'B-day','I-day',
    'B-type','I-type',
    'B-arrive','I-arrive',
    'B-people','I-people',
    'B-depart','I-depart',
    'B-stars','I-stars',
    'B-stay', 'I-stay',
    'B-name', 'I-name',
    'B-dest','I-dest', 
    'B-area','I-area']


slot2desp = {
        'stay':'stay',
    'food':'food',
    'time':'time',
    'type' : 'type',
    'stars':'stars',
    'depart':'depart',
    'area':'area',
    'dest':'dest',
    'people':'people',
    'name':'name',
    'price':'price',
    'arrive':'arrive',
    'day':'day',
    'leave':'leave'
}
domain_set = ["train", "taxi", "restaurant", "attraction", "hotel"]

domain2slot = {
    'train': ['depart','day','people','leave','arrive','dest'],
    'taxi': ['depart','dest','leave','arrive'],
    'hotel': ['day','price','area','people','name','stay','type','stars'],
    'restaurant' : ['day','food','price','area','people','name','time'],
    'attraction' : ['area','name','type']
}

coarse = ['pad', 'O', 'person', 'location', 'special_name', 'common_name', 'number', 'direction', 'others']
bins_labels = ['pad', 'O', 'B-person','I-person' , 'B-location', 'I-location', 'B-special_name', 'I-special_name', 'B-common_name','I-common_name', 'B-number','I-number', 'B-direction','I-direction', 'B-others','I-others']

father_son_slot={
    'pad':['<PAD>'],
    'O':['O'],
    'person':[],
    'location':['name','depart','dest'],
    'special_name':['food'],
    'common_name':['type','price'],
    'number':['stay','stars','leave','arrive','time','day','people'],
    'direction':['area'],
    'others':[]
}

# # bert reps cluster 5
# father_son_slot = {
# 'pad':['<PAD>'],
# 'O':['O'],    
# 'A': ['entity_name', 'playlist', 'artist', 'city', 'party_size_description', 'served_dish', 'poi', 'restaurant_name', 'album', 'track', 'object_name', 'movie_name'], 
# 'B': ['playlist_owner', 'music_item', 'party_size_number', 'state', 'spatial_relation', 'current_location', 'condition_temperature', 'year', 'genre', 'object_select', 'rating_value', 'object_part_of_series_type'], 
# 'C': ['restaurant_type', 'sort', 'cuisine', 'facility', 'condition_description', 'service', 'object_type', 'movie_type', 'location_name'], 
# 'D': ['timeRange', 'country', 'geographic_poi'], 
# 'E': ['best_rating', 'rating_unit', 'object_location_type']
# }
# coarse = ['pad', 'O', 'A', 'B', 'C', 'D', 'E']
# bins_labels = ['pad', 'O', 'B-A','I-A' , 'B-B', 'I-B', 'B-C', 'I-C', 'B-D','I-D', 'B-E','I-E']



SLOT_PAD = 0
PAD_INDEX = 0
UNK_INDEX = 1


class Vocab():
    def __init__(self):
        self.word2index = {"PAD":PAD_INDEX, "UNK":UNK_INDEX}
        self.word2count = {}
        self.index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK"}
        self.n_words = 2
    def index_words(self, sentence):
        for word in sentence:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words+=1
            else:
                self.word2count[word]+=1

def get_father_slot():
    res_dict = {}
    for k,v in father_son_slot.items():
        for s in v:
            res_dict[s] = k
    return res_dict

def read_file(filepath, vocab, son_to_fa_slot, domain=None):
    utter_list, y0_list, y1_list = [], [], []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            splits = line.split("\t")
            tokens = splits[0].split()
            l1_list = splits[1].split()

            utter_list.append(tokens)
            y1_list.append(l1_list)

            vocab.index_words(tokens)

            l0_list = []
            for l in l1_list:
                if "B" in l:
                    l0_list.append("B-"+son_to_fa_slot[l[2:]])
                elif "I" in l:
                    l0_list.append("I-"+son_to_fa_slot[l[2:]])
                else:
                    l0_list.append("O")
            y0_list.append(l0_list)


    data_dict = {"utter": utter_list,"y0":y0_list, "y1": y1_list}
 
    return data_dict, vocab


def read_file2(fpath):
    raw_data = {}
    for intent in domain_set:
        raw_data[intent] = []
        for i, line in enumerate(open(fpath+'/'+intent+'/'+intent+'.txt')):
            temp = []
            tokens, labels = line.strip().split('\t')
            tokens = tokens.split()
            label_list = labels.split()
            if '������' in tokens:
                continue
            temp.append(tokens)
            temp.append(label_list)
            temp.append(intent)
            raw_data[intent].append(temp)
    
    return raw_data

def binarize_data(data, vocab, dm, tag2idx):
    data_bin = {"utter": [],"y0":[], "y1": [],  "domains": []}
    assert len(data_bin["utter"]) == len(data_bin["y0"]) == len(data_bin["y0"])
    dm_idx = domain_set.index(dm)
    for utter_tokens, y0_list, y1_list in zip(data['utter'], data['y0'], data['y1']):
        utter_bin, y0_bin, y1_bin = [], [], []
        for token in utter_tokens:
            utter_bin.append(vocab.word2index[token])
        data_bin['utter'].append(utter_bin)

        for y0 in y0_list:
            y0_bin.append(bins_labels.index(y0))
        data_bin['y0'].append(y0_bin)

        for y1 in y1_list:
            y1_bin.append(tag2idx[y1])
        data_bin['y1'].append(y1_bin)



        assert len(utter_bin) == len(y1_bin) == len(y0_bin)

        data_bin['domains'].append(dm_idx)

  
    
    return data_bin

def datareader(tgt_domain, train_tag2idx, dev_test_tag2idx,  prefix_path='coachdata/'):

    data = {"train": {}, "taxi": {}, "restaurant": {}, "attraction": {}, "hotel": {}}

    # load data and build vocab
    vocab = Vocab()
    
    son_to_fa_slot = get_father_slot()





    trainData, vocab = read_file(prefix_path+"multiwoz/train/train.txt", vocab, son_to_fa_slot, domain="train")
    taxiData, vocab = read_file(prefix_path+"multiwoz/taxi/taxi.txt", vocab, son_to_fa_slot, domain="taxi")
    restaurantData, vocab = read_file(prefix_path+"multiwoz/restaurant/restaurant.txt", vocab, son_to_fa_slot, domain="restaurant")
    attractionData, vocab = read_file(prefix_path+"multiwoz/attraction/attraction.txt", vocab, son_to_fa_slot, domain="attraction")
    hotelData, vocab = read_file(prefix_path+"multiwoz/hotel/hotel.txt", vocab, son_to_fa_slot, domain="hotel")

    
    # binarize data
    data["train"] = binarize_data(trainData, vocab, "train", train_tag2idx if tgt_domain != 'train' else dev_test_tag2idx)
    data["taxi"] = binarize_data(taxiData, vocab, "taxi", train_tag2idx if tgt_domain != 'taxi' else dev_test_tag2idx)
    data["restaurant"] = binarize_data(restaurantData, vocab, "restaurant", train_tag2idx if tgt_domain != 'restaurant' else dev_test_tag2idx)
    data["attraction"] = binarize_data(attractionData, vocab, "attraction", train_tag2idx if tgt_domain != 'attraction' else dev_test_tag2idx)
    data["hotel"] = binarize_data(hotelData, vocab, "hotel", train_tag2idx if tgt_domain != 'hotel' else dev_test_tag2idx)
     
    # print(data['AddToPlaylist'])
    
    return data, vocab

class Dataset(Dataset):
    def __init__(self, X, y0, y1, domains):
        self.X = X
        self.y0 = y0
        self.y1 = y1
        self.domains = domains

    def __getitem__(self, index):

        return self.X[index], self.y0[index], self.y1[index], self.domains[index]
    
    def __len__(self):
        return len(self.X)

def collate_fn(data):
    X, y0, y1, domains = zip(*data)

    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    padded_y0 = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    padded_y1 = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)

    for i, seq in enumerate(y0):
        length = lengths[i]
        padded_y0[i, :length] = torch.LongTensor(seq)

    for i, seq in enumerate(y1):
        length = lengths[i]
        padded_y1[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    domains = torch.LongTensor(domains)

    return padded_seqs, lengths, padded_y0, padded_y1, domains

def get_dataloader(tgt_domain, batch_size, n_samples):

    train_tag2idx = {'O':0}
    dev_test_tag2idx = {'O':0}
    raw_data = read_file2('/home/shenhao/data/coachdata/multiwoz')
    train_data = []
    dev_data = []
    test_data = []
    for k, v in domain2slot.items():
        if k == tgt_domain:
            test_data.extend(raw_data[k])
            for slot in v:
                _B = "B-" + slot
                if _B not in dev_test_tag2idx.keys() and _B in y_set:
                    dev_test_tag2idx[_B] = len(dev_test_tag2idx)
                _I = "I-" + slot
                if _I not in dev_test_tag2idx.keys() and _I in y_set:
                    dev_test_tag2idx[_I] = len(dev_test_tag2idx)
        else:
            train_data.extend(raw_data[k])
            for slot in v:
                _B = "B-" + slot
                if _B not in train_tag2idx.keys() and _B in y_set:
                    train_tag2idx[_B] = len(train_tag2idx)
                _I = "I-" + slot
                if _I not in train_tag2idx.keys() and _I in y_set:
                    train_tag2idx[_I] = len(train_tag2idx)


    train_mask = get_mask_matrix(train_tag2idx, istest=False)
    test_mask = get_mask_matrix(dev_test_tag2idx, istest=True)

    all_data, vocab = datareader(tgt_domain = tgt_domain, train_tag2idx=train_tag2idx, \
        dev_test_tag2idx=dev_test_tag2idx, prefix_path='/home/shenhao/data/coachdata/')

    train_data = {"utter": [], "y0":[], "y1": [], "domains": []}
    for dm_name, dm_data in all_data.items():
        if dm_name != tgt_domain:
            train_data["utter"].extend(dm_data["utter"])
            train_data["y0"].extend(dm_data["y0"])            
            train_data["y1"].extend(dm_data["y1"])

            train_data["domains"].extend(dm_data["domains"])





    val_data = {"utter": [],"y0":[], "y1": [], "y2": [], "domains": []}
    test_data = {"utter": [],"y0":[], "y1": [], "y2": [], "domains": []}
    if n_samples == 0:
        # first 500 samples as validation set
        val_data["utter"] = all_data[tgt_domain]["utter"][:500]  
        val_data["y0"] = all_data[tgt_domain]["y0"][:500]
        val_data["y1"] = all_data[tgt_domain]["y1"][:500]
        val_data["domains"] = all_data[tgt_domain]["domains"][:500]

        # the rest as test set
        test_data["utter"] = all_data[tgt_domain]["utter"][500:]    
        test_data["y0"] = all_data[tgt_domain]["y0"][500:]
        test_data["y1"] = all_data[tgt_domain]["y1"][500:]      # rest as test set
        test_data["domains"] = all_data[tgt_domain]["domains"][500:]    # rest as test set

    else:
        # first n samples as train set
        train_data["utter"].extend(all_data[tgt_domain]["utter"][:n_samples])
        train_data["y0"].extend(all_data[tgt_domain]["y0"][:n_samples])
        train_data["y1"].extend(all_data[tgt_domain]["y1"][:n_samples])
        train_data["domains"].extend(all_data[tgt_domain]["domains"][:n_samples])


        # from n to 500 samples as validation set
        val_data["utter"] = all_data[tgt_domain]["utter"][n_samples:500]  
        val_data["y0"] = all_data[tgt_domain]["y0"][n_samples:500]
        val_data["y1"] = all_data[tgt_domain]["y1"][n_samples:500]
        val_data["domains"] = all_data[tgt_domain]["domains"][n_samples:500]

        # the rest as test set (same as zero-shot)
        test_data["utter"] = all_data[tgt_domain]["utter"][500:]
        test_data["y0"] = all_data[tgt_domain]["y0"][500:]
        test_data["y1"] = all_data[tgt_domain]["y1"][500:]
        test_data["domains"] = all_data[tgt_domain]["domains"][500:]


    dataset_tr = Dataset(train_data["utter"], train_data["y0"], train_data["y1"],train_data["domains"])
    dataset_val = Dataset(val_data["utter"], val_data["y0"],val_data["y1"], val_data["domains"])
    dataset_test = Dataset(test_data["utter"], test_data["y0"], test_data["y1"], test_data["domains"])

    
    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn= collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    

    return dataloader_tr, dataloader_val, dataloader_test, vocab, train_tag2idx, dev_test_tag2idx, train_mask, test_mask

def get_mask_matrix(son_dict, istest):

    fine2coarse = {}

    for k,v in father_son_slot.items():
            for t in v:
                fine2coarse[t] = k

    id2tag = {v:k for k,v in son_dict.items()}
    vec_len = len(bins_labels)
    mask_list = np.zeros((len(son_dict), vec_len))
    for i in range(len(id2tag)):
        tag = id2tag[i]
        if tag != 'O':
                


            
            newtag = tag[:2] + fine2coarse[tag[2:]]
            idx = bins_labels.index(newtag)
            mask_list[i][idx] = 1
            # if istest:
            #     if tag not in ['B-object_type', 'I-object_type']:
            #         mask_list[i][idx] = 1
            #     elif tag in ['B-object_type', 'I-object_type']:
            #         mask_list[i] += 0.05
            #         idxt = bins_labels.index('B-common_name')
            #         mask_list[i][idxt] = 0
            #         idxt = bins_labels.index('I-common_name')
            #         mask_list[i][idxt] = 0


            
        elif tag == 'O':
            idx = bins_labels.index(tag)
            mask_list[i][idx] = 1

    mask_list = mask_list.transpose(0, 1)
    return mask_list
# a,b,c,vocab,e,f = get_dataloader('AddToPlaylist', 4, 0)
# embeddings = load_embedding(vocab, 300, '/home/sh/data/glove.6B.300d.txt')
# print(embeddings.shape)

