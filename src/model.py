import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import torchtext
import random
from collections import Counter
from .datareader import slot2desp

class LabelEmbeddingFactory:
    def __init__(self):
        pass

    def BertEncoderAve(self, tag2idx, tokenizer, encoder):
        emb = []
        dim = 768
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
                tokens = tokenizer.encode(slot)
                tokens = torch.tensor(tokens)
                tokens = tokens.unsqueeze(0)
                reps = encoder(tokens)[0].mean(1).squeeze()
                reps = reps.detach().cpu().numpy()
                if tag[0] == "B":
                    reps = np.concatenate((np.array([1,0,0], dtype=np.float32), reps), 0)
                    


                elif tag[0] == "I":
                    reps = np.concatenate((np.array([0,1,0], dtype=np.float32), reps), 0)
                

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

class CRF:
    pass

class ProjMartrix(nn.Module):
    def __init__(self, params, dataloader_tr):
        super(ProjMartrix, self).__init__()
        self.params = params
        self.tokenizer = BertTokenizer.from_pretrained(params.bert_path)
        self.encoder = BertModel.from_pretrained(params.bert_path)

        self.Proj = nn.Parameter(torch.empty(768, 300))
        nn.init.xavier_normal_(self.Proj, -0.1, 0.1)
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
        c = [(k, v) for k, v in c.items()][1:3000]
        
        # 加入高频词
        for k, v in c:
            tokens.append(k)
        
        加入所有训练语料token
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

class SlotFilling(nn.Module):
    def __init__(self, params, train_tag2idx, dev_test_tag2idx, bert_path, device):
        super(SlotFilling, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.encoder = BertModel.from_pretrained(bert_path)
        self.device = device
        self.params = params
        labelembedding = LabelEmbeddingFactory()
        if params.emb_src == 'Bert':
            self.train_labelembedding = labelembedding.BertEncoderAve(train_tag2idx,self.tokenizer,self.encoder).to(device)
            self.dev_test_labelembedding = labelembedding.BertEncoderAve(dev_test_tag2idx, self.tokenizer ,self.encoder).to(device)
        elif params.emb_src == 'Glove':
            self.train_labelembedding = labelembedding.GloveEmbAve(params.emb_file, train_tag2idx).to(device)
            self.dev_test_labelembedding = labelembedding.GloveEmbAve(params.emb_file, dev_test_tag2idx).to(device)
        self.sim_func = Similarity(size=(768, 768 + 3), type='mul')
        self.sim_func2 = Similarity(size=(768, 300 + 3), type='mul')
        self.Proj_W = nn.Parameter(torch.empty(768, 300))
        torch.nn.init.uniform_(self.Proj_W, -0.1, 0.1)

    def forward(self, x, heads, seq_len, domains, iseval=False, y=None):
        
        attention_mask = (x != 0).byte().to(self.device)
        reps = self.encoder(x, attention_mask=attention_mask)[0]
        labelembedding = None
        if not iseval:
            labelembedding = self.train_labelembedding
        else:
            labelembedding = self.dev_test_labelembedding

        if self.params.proj == 'no':
            score = self.sim_func(reps, labelembedding)
        else:
            prefix ,suffix = torch.split(labelembedding, [3, 768], dim=-1)
            suffix_embedding = torch.matmul(suffix, self.Proj_W)
            labelembedding = torch.cat((prefix, suffix_embedding), dim=-1)
            score = self.sim_func2(reps, labelembedding)

        
        return score    

