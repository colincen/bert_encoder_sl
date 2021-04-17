import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import torchtext
import random
from collections import Counter
from .datareader import slot2desp, father_son_slot, coarse
import torch.nn.functional as F
from .crf import CRF
from .crf_labelembedding import CRF as CRF_labelembedding

class LabelEmbeddingFactory:
    def __init__(self):
        self.fine2coarse = {}
        for k,v in father_son_slot.items():
            for t in v:
                self.fine2coarse[t] = k
        self.coarse2vec = {}
        self.coarse2vec['pad'] = np.zeros(9, dtype = np.float32)
        self.coarse2vec['O'] = np.zeros(9, dtype = np.float32)
        self.coarse2vec['person'] = np.zeros(9, dtype = np.float32)
        self.coarse2vec['location'] = np.zeros(9, dtype = np.float32)
        self.coarse2vec['special_name'] = np.zeros(9, dtype = np.float32)
        self.coarse2vec['common_name'] = np.zeros(9, dtype = np.float32)
        self.coarse2vec['number'] = np.zeros(9, dtype = np.float32)
        self.coarse2vec['direction'] = np.zeros(9, dtype = np.float32)
        self.coarse2vec['others'] = np.zeros(9, dtype = np.float32)

        cnt = 0
        for k,v in self.coarse2vec.items():
            if k != 'pad' :
                self.coarse2vec[k][cnt] = 1
                cnt += 1
        print(self.coarse2vec)

    def BertEncoderAve(self, tag2idx, tokenizer, encoder):
        emb = []
        dim = 768
        idx2tag = {v:k for k, v in tag2idx.items()}
        tag_idx_pair = [(i, idx2tag[i]) for i in range(len(idx2tag))]
        for idx, tag in tag_idx_pair:
            tag_emb = None
            ####################
            if tag == "<PAD>":
                # tag_emb = np.zeros(dim+3+9, dtype=np.float32)
                tag_emb = np.concatenate((np.array([0,0,0],dtype = np.float32), 
                self.coarse2vec[self.fine2coarse[tag]], 
                np.zeros(dim, dtype=np.float32)), 0)

            elif tag == "O":
                # tag_emb = np.zeros(dim+3+9, dtype=np.float32)
                # tag_emb[2] = 1
                tag_emb = np.concatenate((np.array([0,0,1], dtype=np.float32), 
                self.coarse2vec[self.fine2coarse[tag]], 
                np.zeros(dim, dtype=np.float32)), 0)

            ####################
            else:
                slot = tag[2:]
                #########
                # 这里可以把slot改成description试试
                tokens = tokenizer.encode(slot2desp[slot])
                #########
                tokens = torch.tensor(tokens)
                tokens = tokens.unsqueeze(0)
                reps = encoder(tokens)[0].mean(1).squeeze()
                reps = reps.detach().cpu().numpy()
                if tag[0] == "B":
                    reps = np.concatenate((np.array([1,0,0], dtype=np.float32), 
                    self.coarse2vec[self.fine2coarse[tag[2:]]],
                    reps), 0)
                    


                elif tag[0] == "I":
                    reps = np.concatenate((np.array([0,1,0], dtype=np.float32), 
                    self.coarse2vec[self.fine2coarse[tag[2:]]],
                    reps), 0)
                

                tag_emb = reps

            emb.append(tag_emb)
        
        return torch.tensor(emb)

    def GloveEmbAve(self, emb_path, tag2idx):
        glove = torchtext.vocab.GloVe(cache=emb_path, name='6B')
        emb = []
        dim = 300
        idx2tag = {v:k for k, v in tag2idx.items()}
        tag_idx_pair = [(i, idx2tag[i]) for i in range(len(idx2tag))]
        for idx, tag in tag_idx_pair:
            tag_emb = None
            ####################
            if tag == "<PAD>":
                tag_emb = np.zeros(dim+3, dtype=np.float32)
                
            elif tag == "O":
                tag_emb = np.zeros(dim+3, dtype=np.float32)
                tag_emb[2] = 1
            ####################
            else:
                slot = tag[2:]
                desp = slot2desp[slot].split(' ')
                slot_tokens = []
                for i in desp:
                    slot_tokens.append(np.reshape(glove[i], (300, 1)))
                slot_tokens = np.concatenate(slot_tokens, -1)
                slot_tokens = np.mean(slot_tokens, -1)

                if tag[0] == "B":
                    reps = np.concatenate((np.array([1,0,0], dtype=np.float32), slot_tokens), 0)
                    


                elif tag[0] == "I":
                    reps = np.concatenate((np.array([0,1,0], dtype=np.float32), slot_tokens), 0)
                

                tag_emb = reps

            emb.append(tag_emb)
        
        return torch.tensor(emb)

class Similarity(nn.Module):
    def __init__(self, size, type='mul'):
        super(Similarity, self).__init__()
        output_dim, emb_dim = size
        self.W = nn.Parameter(torch.randn(output_dim, emb_dim))
        nn.init.xavier_normal_(self.W)
        self.type = type
    
    def forward(self, output_reps, emb):
        if self.type == 'mul':
            reps = torch.matmul(output_reps, self.W)
            reps = torch.matmul(reps, emb.transpose(0,1))
            
            return reps


class SlotFilling(nn.Module):
    def __init__(self, params, train_tag2idx, dev_test_tag2idx, bert_path, device):
        super(SlotFilling, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.encoder = BertModel.from_pretrained(bert_path)
        self.device = device
        self.params = params
        self.crf = CRF(num_tags=16, batch_first=True)
        labelembedding = LabelEmbeddingFactory()

        self.train_labelembedding = labelembedding.BertEncoderAve(train_tag2idx,self.tokenizer,self.encoder).to(device)
    
        self.dev_test_labelembedding = labelembedding.BertEncoderAve(dev_test_tag2idx, self.tokenizer ,self.encoder).to(device)
    
        self.crf_labemb = CRF_labelembedding(train_labelEmbedding=self.train_labelembedding, \
                                                dev_test_labelEmbedding=self.dev_test_labelembedding, \
                                            
                                                 train_num_tags=self.train_labelembedding.size(0), \
                                                     dev_test_num_tags = self.dev_test_labelembedding.size(0),\
                                                     batch_first = True)

        
        self.sim_func = Similarity(size=(768, 768 + 3 + 9), type='mul')
        self.sim_func2 = Similarity(size=(768, 300 + 3), type='mul')
        self.Proj_W = nn.Parameter(torch.empty(768, 300))
        self.droupout = nn.Dropout(params.dropout)
        torch.nn.init.uniform_(self.Proj_W, -0.1, 0.1)


        self.coarse_emb = nn.Linear(768, 300)
        self.fc_for_coarse = nn.Linear(300, 16)

        self.fine_emb = nn.Linear(768, 468)


        # self.fc_for_concat_emb = nn.Linear(768 * 2, 768)






    def forward(self, x, heads, seq_len, domains, iseval=False, y=None, bin_tag=None):
        
        attention_mask = (x != 0).byte().to(self.device)
        reps = self.encoder(x, attention_mask=attention_mask)[0]
        bert_out_reps = self.droupout(reps)
        labelembedding = None
        if not iseval:
            labelembedding = self.train_labelembedding
            
        else:
            labelembedding = self.dev_test_labelembedding

        reps = bert_out_reps
        coarse_reps = self.coarse_emb(reps)
        coarse_logits = self.fc_for_coarse(coarse_reps)
        
        if not iseval:
            coarse_loss = -self.crf(emissions=coarse_logits, tags=bin_tag,
            mask=attention_mask,reduction='mean')
            reps = self.fine_emb(reps)
            reps = torch.cat((coarse_reps, reps), -1)
            logits = self.sim_func(reps, labelembedding)
            emb_loss = -self.crf_labemb(logits, y, attention_mask, 'mean')

        else:
            coarse_loss = torch.tensor(0, device=self.device)
            emb_loss = torch.tensor(0, device=self.device)
            reps = self.fine_emb(reps)
            reps = torch.cat((coarse_reps, reps), -1)
            logits = self.sim_func(reps, labelembedding)



        # reps = self.fine_emb(reps)

        # reps = self.fc_for_concat_emb(torch.cat((coarse_reps, reps), -1))

        # if self.params.proj == 'no':
        #     final_logits = self.sim_func(reps, labelembedding)


        
        return coarse_logits, logits, bert_out_reps, reps, coarse_loss, emb_loss

class ProjMartrix(nn.Module):
    def __init__(self, params, dataloader_tr):
        super(ProjMartrix, self).__init__()
        self.params = params
        self.tokenizer = BertTokenizer.from_pretrained(params.bert_path)
        self.encoder = BertModel.from_pretrained(params.bert_path)

        self.Proj = nn.Parameter(torch.empty(768, 300), requires_grad=False)
        self.dropout = nn.Dropout(params.dropout)
        nn.init.xavier_normal_(self.Proj)
        with open(params.corps_path, 'r') as f:
            self.corps = f.read()
        f.close()
        self.train_tokens = []
        self.train_labels = []
        for words, x, is_heads, pad_heads, tags, y, domains, seq_len in dataloader_tr:
            for word in words:
                self.train_tokens.extend(word.split(' '))
            for tag in tags:
                self.train_labels.extend(tag.split(' '))
        



    def forward(self, x, y):
        inputs = self.tokenizer.batch_encode_plus(x,
                                    pad_to_max_length=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_ids = torch.tensor(input_ids, device=self.params.device)
        attention_mask = torch.tensor(attention_mask, device=self.params.device)

        output = self.encoder(input_ids, attention_mask=attention_mask)[0]
        output = output.mean(1)
        output = torch.matmul(output, self.Proj)
        output = self.dropout(output)
        return output


    @staticmethod
    def words2vec(word2emb, input_toks):
        
        
        emb = None

        if isinstance(input_toks, str):
            if input_toks in word2emb.itos:
                
                return [input_toks], word2emb[input_toks]
        else:
            temp = []

            for i in input_toks:
                if i not in word2emb.itos:
                    return None, None
                # print(i)
                # print(np.reshape(word2emb[i], (300, 1)))

                temp.append(np.reshape(word2emb[i], (300, 1)))
            emb = np.concatenate(temp, -1)
            emb = np.mean(emb, -1)

            return input_toks, emb


        return None, None
            
    
    def get_words(self, emb_path):
        tokens = []
        c = Counter(self.corps.split(' '))
        c = [(k, v) for k, v in c.items()][1:10000]
        
        # 加入高频词
        for k, v in c:
            tokens.append(k)
        
        # 加入所有训练语料token
        for w in self.train_tokens:
            if w not in tokens:
                tokens.append(w)

        # 加入所有槽名
        for k,v in slot2desp.items():
            v = v.split(' ')
            for j in v:
                if j not in tokens:
                    tokens.append(j)

        # 加入 token span
        i = 0
        data = self.train_tokens
        label = self.train_labels
        while i < len(data):
            fg = False
            if label[i][0] == 'B':
                fg = True
                start = i
                end = i + 1
                while end < len(data) and label[end][0] == 'I':
                    end += 1
                end = min(end, len(data))
                slotName = label[i][2:]
                temp = []
                # if slotName not in slot2exemplar:
                #     slot2exemplar[slotName] = []
                for j in range(start, end):
                    temp.append(data[j])
                tokens.append(temp)
                i = end
            if not fg:
                i += 1
        
        glove = torchtext.vocab.GloVe(cache=emb_path, name='6B')

        pair_x = []
        pair_y = []
        for tok in tokens:
            toks, emb = ProjMartrix.words2vec(glove, tok)
            if toks is not None:
                pair_x.append(toks)
                pair_y.append(emb)
        return pair_x, pair_y


    def batch_generator(self, pair_x, pair_y):
        assert len(pair_x) == len(pair_y)
        data = [(pair_x[i], pair_y[i]) for i in range(len(pair_x))]
        random.shuffle(data)
        batch_size = 32
        b = int(len(data) / batch_size)
        for i in range(b):
            batch = data[i*batch_size : min(len(data), (i+1)*batch_size)]
            x = [ba[0] for ba in batch]
            y = [ba[1].reshape(1, 300) for ba in batch]
            y = np.concatenate(y, 0)


            x = [' '.join(j) for j in x]

            yield x,y
