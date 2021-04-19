
from transformers import BertTokenizer
import torch

from torch.utils.data import DataLoader, Dataset
import numpy as np
slot_list = ['<PAD>','playlist', 'music_item', 'geographic_poi', 'facility', 
'movie_name', 'location_name', 'restaurant_name', 'track', 'restaurant_type', 
'object_part_of_series_type', 'country', 'service', 'poi', 'party_size_description',
'served_dish', 'genre', 'current_location', 'object_select', 'album', 'object_name',
'state', 'sort', 'object_location_type', 'movie_type', 'spatial_relation', 'artist', 
'cuisine', 'entity_name', 'object_type', 'playlist_owner', 'timeRange', 'city',
'rating_value', 'best_rating', 'rating_unit', 'year', 'party_size_number',
'condition_description', 'condition_temperature']

y_set = ['<PAD>' ,'O', 'B-playlist', 'I-playlist', 'B-music_item', 'I-music_item', 'B-geographic_poi', 
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

coarse = ['pad', 'O', 'person', 'location', 'special_name', 'common_name', 'number', 'direction', 'others']
bins_labels = ['pad', 'O', 'B-person','I-person' , 'B-location', 'I-location', 'B-special_name', 'I-special_name', 'B-common_name','I-common_name', 'B-number','I-number', 'B-direction','I-direction', 'B-others','I-others']

father_son_slot={
    'pad':['<PAD>'],
    'O':['O'],
    'person':['artist','playlist_owner','party_size_description'],
    'location':['state','city','geographic_poi','object_location_type','location_name','country','poi'],
    'special_name':['album','service','entity_name','playlist','music_item','track','movie_name','object_name',
                    'served_dish','restaurant_name','cuisine'],
    'common_name':['object_type', 'object_part_of_series_type','movie_type','restaurant_type','genre','facility',
                'condition_description','condition_temperature'],
    'number':['rating_value','best_rating','year','party_size_number','timeRange'],
    'direction':['spatial_relation','current_location','object_select'],
    'others':['rating_unit', 'sort']
}



class NerDataset(Dataset):
    def __init__(self, raw_data, tag2idx ,bert_path):
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        sents, tags, domains = [], [], []
        for entry in raw_data:
            sents.append(["[CLS]"] + entry[0] + ["[SEP]"])
            tags.append(["<PAD>"] + entry[1] + ["<PAD>"])
            domains.append([entry[-1]])
        self.fine2coarse = {}
        for k,v in father_son_slot.items():
            for t in v:
                self.fine2coarse[t] = k

        self.sents, self.tags, self.domains, self.tag2idx = sents, tags, domains, tag2idx
    
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx):
        tag2idx = self.tag2idx
        words, tags, domains = self.sents[idx], self.tags[idx], self.domains[idx]



        domains = domain_set.index(domains[0])

        x, y = [], []
        coarse_labels = []
        bin_tags = []
        is_heads = []
        for w, t in zip(words, tags):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)
            is_head = [1] + [0] * (len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)

            yy = [tag2idx[each] for each in t]

            coarse_label = []
            bin_tag = []

            for lab in t:
                if lab in ['O','<PAD>']:
                    coarse_label.append(coarse.index(self.fine2coarse[lab]))
                    bin_tag.append(bins_labels.index(self.fine2coarse[lab]))

                else:
                    slot = lab[2:]
                    coarse_label.append(coarse.index(self.fine2coarse[slot]))
                    bin_tag.append(bins_labels.index(lab[:2] + self.fine2coarse[slot]))
            

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
            coarse_labels.extend(coarse_label)
            bin_tags.extend(bin_tag)



        assert len(x) == len(y) == len(is_heads) == len(coarse_labels) == len(bin_tags)



        seq_len = len(y)
        

        words = " ".join(words)
        tags = " ".join(tags)

        return words, x, is_heads, tags, y, domains, seq_len, coarse_labels, bin_tags

def read_file(fpath):
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


def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(6)
    domains = f(5)
    coarse_labels = f(7)
    bin_tags = f(8)
    maxlen = np.array(seqlens).max()


    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(4, maxlen)
    heads = f(2 ,maxlen)
    pad_coarse_labels = f(7, maxlen)
    pad_bin_tags = f(8, maxlen)

    f = torch.LongTensor

    return words, f(x), is_heads, f(heads), tags, f(y), domains, seqlens, f(pad_coarse_labels), f(pad_bin_tags)


def get_dataloader(tgt_domain, batch_size, fpath, bert_path, n_samples=0):

    raw_data = read_file(fpath)
    train_tag2idx = {"<PAD>" : 0, "O":1}
    dev_test_tag2idx = {"<PAD>" : 0, "O":1}
    train_data = []
    dev_data = []
    test_data = []

    for k, v in domain2slot.items():
        if k == tgt_domain:
            test_data.extend(raw_data[k][n_samples:])
            for slot in v:
                _B = "B-" + slot
                if _B not in dev_test_tag2idx.keys() and _B in y_set:
                    dev_test_tag2idx[_B] = len(dev_test_tag2idx)
                _I = "I-" + slot
                if _I not in dev_test_tag2idx.keys() and _I in y_set:
                    dev_test_tag2idx[_I] = len(dev_test_tag2idx)
            
            if n_samples > 0:
                train_data.extend(raw_data[k][:n_samples])    
                

                for slot in v:
                    _B = "B-" + slot
                    if _B not in train_tag2idx.keys() and _B in y_set:
                        train_tag2idx[_B] = len(train_tag2idx)
                    _I = "I-" + slot
                    if _I not in train_tag2idx.keys() and _I in y_set:
                        train_tag2idx[_I] = len(train_tag2idx)

        else:
            train_data.extend(raw_data[k])
            for slot in v:
                _B = "B-" + slot
                if _B not in train_tag2idx.keys() and _B in y_set:
                    train_tag2idx[_B] = len(train_tag2idx)
                _I = "I-" + slot
                if _I not in train_tag2idx.keys() and _I in y_set:
                    train_tag2idx[_I] = len(train_tag2idx)
    


        fine2coarse = {}
        for k,v in father_son_slot.items():
            for t in v:
                fine2coarse[t] = k
    

    train_mask = get_mask_matrix(train_tag2idx)
    test_mask = get_mask_matrix(dev_test_tag2idx)

    
    dev_data = test_data[n_samples:500]
    test_data = test_data[500:]

    dataset_tr = NerDataset(raw_data=train_data, tag2idx=train_tag2idx, bert_path=bert_path)
    dataset_val = NerDataset(raw_data=dev_data, tag2idx=dev_test_tag2idx, bert_path=bert_path)    
    dataset_test = NerDataset(raw_data=test_data, tag2idx=dev_test_tag2idx, bert_path=bert_path)    
   
    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=pad)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=pad)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=pad)

    return dataloader_tr, dataloader_val, dataloader_test, train_tag2idx, dev_test_tag2idx, train_mask,  test_mask


def get_mask_matrix(son_dict):

    fine2coarse = {}

    for k,v in father_son_slot.items():
            for t in v:
                fine2coarse[t] = k

    id2tag = {v:k for k,v in son_dict.items()}
    vec_len = len(bins_labels)
    mask_list = np.zeros((len(son_dict), vec_len))
    for i in range(len(id2tag)):
        tag = id2tag[i]
        if tag not in ['<PAD>', 'O']:
            newtag = tag[:2] + fine2coarse[tag[2:]]
            idx = bins_labels.index(newtag)
            mask_list[i][idx] = 1
        elif tag == '<PAD>':
            idx = bins_labels.index('pad')
            mask_list[i][idx] = 1
        elif tag == 'O':
            idx = bins_labels.index(tag)
            mask_list[i][idx] = 1

    mask_list = mask_list.transpose(0, 1)
    return mask_list
    





