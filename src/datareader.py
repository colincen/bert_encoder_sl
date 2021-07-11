
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

# father_son_slot={
#     'pad':['<PAD>'],
#     'O':['O'],
#     'person':[],
#     'location':['name','depart','dest'],
#     'special_name':['food'],
#     'common_name':['type','price'],
#     'number':['stay','stars','leave','arrive','time','day','people'],
#     'direction':['area'],
#     'others':[]
# }
father_son_slot={
    'pad':['<PAD>'],
    'O':['O'],
    'A': ['dest', 'depart', 'day', 'price', 'type', 'area', 'name', 'food'],
     'B': ['arrive', 'leave', 'time'],
      'C': ['people', 'stay', 'stars']
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
    

    train_mask = get_mask_matrix(train_tag2idx, istest=False)
    test_mask = get_mask_matrix(dev_test_tag2idx, istest=True)

    
    dev_data = test_data[n_samples:500]
    test_data = test_data[500:]

    dataset_tr = NerDataset(raw_data=train_data, tag2idx=train_tag2idx, bert_path=bert_path)
    dataset_val = NerDataset(raw_data=dev_data, tag2idx=dev_test_tag2idx, bert_path=bert_path)    
    dataset_test = NerDataset(raw_data=test_data, tag2idx=dev_test_tag2idx, bert_path=bert_path)    
   
    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=pad)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=pad)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=pad)

    return dataloader_tr, dataloader_val, dataloader_test, train_tag2idx, dev_test_tag2idx, train_mask,  test_mask


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
        if tag not in ['<PAD>', 'O']:
                


            
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


            
        elif tag == '<PAD>':
            idx = bins_labels.index('pad')
            mask_list[i][idx] = 1
        elif tag == 'O':
            idx = bins_labels.index(tag)
            mask_list[i][idx] = 1

    mask_list = mask_list.transpose(0, 1)
    return mask_list
    





