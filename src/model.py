import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np

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

class SlotFilling(nn.Module):
    def __init__(self, train_tag2idx, dev_test_tag2idx, bert_path, device):
        super(SlotFilling, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.encoder = BertModel.from_pretrained(bert_path)
        self.device = device
        labelembedding = LabelEmbeddingFactory()
        self.train_labelembedding = labelembedding.BertEncoderAve(train_tag2idx,self.tokenizer,self.encoder).to(device)
        self.dev_test_labelembedding = labelembedding.BertEncoderAve(dev_test_tag2idx, self.tokenizer ,self.encoder).to(device)
        self.sim_func = Similarity(size=(768, 768 + 3), type='mul')

    def forward(self, x, heads, seq_len, domains, iseval=False, y=None):
        
        attention_mask = (x != 0).byte().to(self.device)
        reps = self.encoder(x, attention_mask=attention_mask)[0]
        labelembedding = None
        if not iseval:
            labelembedding = self.train_labelembedding
        else:
            labelembedding = self.dev_test_labelembedding

        score = self.sim_func(reps, labelembedding)
        
        return score    