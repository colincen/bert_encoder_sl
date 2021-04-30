from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
slot_list = ['playlist', 'music_item', 'geographic_poi', 'facility', 
'movie_name', 'location_name', 'restaurant_name', 'track', 'restaurant_type', 
'object_part_of_series_type', 'country', 'service', 'poi', 'party_size_description',
'served_dish', 'genre', 'current_location', 'object_select', 'album', 'object_name',
'state', 'sort', 'object_location_type', 'movie_type', 'spatial_relation', 'artist', 
'cuisine', 'entity_name', 'object_type', 'playlist_owner', 'timeRange', 'city',
'rating_value', 'best_rating', 'rating_unit', 'year', 'party_size_number',
'condition_description', 'condition_temperature']

y_set = ['O', 'B-playlist', 'I-playlist', 'B-music_item', 'I-music_item', 'B-geographic_poi', 
'I-geographic_poi', 'B-facility', 'I-facility', 'B-movie_name', 'I-movie_name', 'B-location_name', 'I-location_name', 
'B-restaurant_name', 'I-restaurant_name', 'B-track', 'I-track', 'B-restaurant_type', 'I-restaurant_type', 
'B-object_part_of_series_type', 'I-object_part_of_series_type', 'B-country', 'I-country', 'B-service', 'I-service',
 'B-poi', 'I-poi', 'B-party_size_description', 'I-party_size_description', 'B-served_dish', 'I-served_dish', 
 'B-genre',  'I-genre', 'B-current_location', 'I-current_location', 'B-object_select', 'I-object_select', 
 'B-album', 'I-album', 'B-object_name', 'I-object_name', 'B-state', 'I-state', 'B-sort', 'I-sort',
  'B-object_location_type', 'I-object_location_type', 'B-movie_type', 'I-movie_type', 'B-spatial_relation', 'I-spatial_relation',
   'B-artist', 'I-artist', 'B-cuisine', 'I-cuisine', 'B-entity_name', 'I-entity_name', 'B-object_type', 'I-object_type', 
   'B-playlist_owner', 'I-playlist_owner', 'B-timeRange', 'I-timeRange', 'B-city', 'I-city', 'B-rating_value',
    'B-best_rating', 'B-rating_unit', 'B-year', 'B-party_size_number', 'B-condition_description', 'B-condition_temperature']


slot2desp = {'playlist': 'playlist',
 'music_item': 'music item',
  'geographic_poi': 'geographic position',
 'facility': 'facility',
#   'movie_name': 'movie',
    'movie_name': 'movie name',
#    'location_name': 'location',
      'location_name': 'location name',
    # 'restaurant_name': 'restaurant',
    'restaurant_name': 'restaurant name',
  'track': 'track',
   'restaurant_type': 'restaurant type',
    'object_part_of_series_type': 'series',
     'country': 'country', 
  'service': 'service',
   'poi': 'position',
    'party_size_description': 'person',
     'served_dish': 'served dish',
      'genre': 'genre', 
  'current_location': 'current location',
   'object_select': 'this current',
    'album': 'album',
    #  'object_name': 'object name',
          'object_name': 'object name',
   'state': 'location', 
   'sort': 'type', 
   'object_location_type': 'location type',
    'movie_type': 'movie type',
    'spatial_relation': 'spatial relation',
     'artist': 'artist', 
     'cuisine': 'cuisine',
      'entity_name': 'entity name',
     'object_type': 'object type',
      'playlist_owner': 'owner',
       'timeRange': 'time range', 
       'city': 'city',
        'rating_value': 'rating value',
        # 'rating_value': 'rating value num',
      'best_rating': 'best rating',
       'rating_unit': 'rating unit',
        'year': 'year', 
        # 'year': 'year num', 
        'party_size_number': 'number', 
      'condition_description': 'weather',
       'condition_temperature': 'temperature'
}
domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather",\
     "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],

    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 
    'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],

    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}



coarse = ['O', 'person', 'location', 'special_name', 'common_name', 'number', 'direction', 'others']
bins_labels = ['O', 'B-person','I-person' , 'B-location', 'I-location', 'B-special_name', 'I-special_name', 'B-common_name','I-common_name', 'B-number','I-number', 'B-direction','I-direction', 'B-others','I-others']

father_son_slot={
    'O' : ['O'],
    'person':['artist','party_size_description'],
    'location':['state','city','geographic_poi','object_location_type','location_name','country','poi'],
    'special_name':['album','service','entity_name','playlist','music_item','track','movie_name','object_name',
                    'served_dish','restaurant_name','cuisine'],
    'common_name':['object_type', 'object_part_of_series_type','movie_type','restaurant_type','genre','facility',
                'condition_description','condition_temperature'],
    'number':['rating_value','best_rating','year','party_size_number','timeRange'],
    'direction':['spatial_relation','current_location','object_select'],
    'others':['rating_unit', 'sort','playlist_owner']
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

    data = {"AddToPlaylist": {}, "BookRestaurant": {}, "GetWeather": {}, "PlayMusic": {}, "RateBook": {}, "SearchCreativeWork": {}, "SearchScreeningEvent": {}}

    # load data and build vocab
    vocab = Vocab()
    
    son_to_fa_slot = get_father_slot()


    AddToPlaylistData, vocab = read_file(prefix_path+"snips/AddToPlaylist/AddToPlaylist.txt", vocab, son_to_fa_slot, domain="AddToPlaylist")
    BookRestaurantData, vocab = read_file(prefix_path+"snips/BookRestaurant/BookRestaurant.txt", vocab, son_to_fa_slot, domain="BookRestaurant")
    GetWeatherData, vocab = read_file(prefix_path+"snips/GetWeather/GetWeather.txt", vocab, son_to_fa_slot, domain="GetWeather")
    PlayMusicData, vocab = read_file(prefix_path+"snips/PlayMusic/PlayMusic.txt", vocab, son_to_fa_slot, domain="PlayMusic")
    RateBookData, vocab = read_file(prefix_path+"snips/RateBook/RateBook.txt", vocab, son_to_fa_slot, domain="RateBook")
    SearchCreativeWorkData, vocab = read_file(prefix_path+"snips/SearchCreativeWork/SearchCreativeWork.txt", vocab, son_to_fa_slot, domain="SearchCreativeWork")
    SearchScreeningEventData, vocab = read_file(prefix_path+"snips/SearchScreeningEvent/SearchScreeningEvent.txt", vocab, son_to_fa_slot, domain="SearchScreeningEvent")

    
    # binarize data
    data["AddToPlaylist"] = binarize_data(AddToPlaylistData, vocab, "AddToPlaylist", train_tag2idx if tgt_domain != 'AddToPlaylist' else dev_test_tag2idx)
    data["BookRestaurant"] = binarize_data(BookRestaurantData, vocab, "BookRestaurant", train_tag2idx if tgt_domain != 'BookRestaurant' else dev_test_tag2idx)
    data["GetWeather"] = binarize_data(GetWeatherData, vocab, "GetWeather", train_tag2idx if tgt_domain != 'GetWeather' else dev_test_tag2idx)
    data["PlayMusic"] = binarize_data(PlayMusicData, vocab, "PlayMusic", train_tag2idx if tgt_domain != 'PlayMusic' else dev_test_tag2idx)
    data["RateBook"] = binarize_data(RateBookData, vocab, "RateBook", train_tag2idx if tgt_domain != 'RateBook' else dev_test_tag2idx)
    data["SearchCreativeWork"] = binarize_data(SearchCreativeWorkData, vocab, "SearchCreativeWork", train_tag2idx if tgt_domain != 'SearchCreativeWork' else dev_test_tag2idx)
    data["SearchScreeningEvent"] = binarize_data(SearchScreeningEventData, vocab, "SearchScreeningEvent",  train_tag2idx if tgt_domain != 'SearchScreeningEvent' else dev_test_tag2idx)
    
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
    raw_data = read_file2('/home/sh/data/coachdata/snips')
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
        dev_test_tag2idx=dev_test_tag2idx, prefix_path='/home/sh/data/coachdata/')

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

