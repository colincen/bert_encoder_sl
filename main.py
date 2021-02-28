import src.model as Model
from src.model import ProjMartrix
import torch
import torch.nn as nn
from src.datareader import get_dataloader
from src.conlleval import evaluate
from config import get_params, init_experiment
from tqdm import tqdm
import numpy as np
import os
import json

class Main:
    def __init__(self, params, logger):
        self.params = params
        self.logger = logger
        self.loss_func = nn.CrossEntropyLoss()
        self.model_saved_path = None
        self.opti_saved_path = None


    def do_pretrain(self):
        dataloader_tr, dataloader_val, dataloader_test, train_tag2idx, dev_test_tag2idx = get_dataloader(params.tgt_domain, batch_size=params.batch_size, fpath=params.file_path, bert_path=params.bert_path)
        premodel = ProjMartrix(params, dataloader_tr)  
        premodel.to(params.device)
        premodel.train()
        pair_x, pair_y = premodel.get_words(params.emb_file)
        pre_loss_func = nn.MSELoss()
        pre_optimizer = torch.optim.Adam(premodel.parameters(), lr = params.lr)
        for i, (x, y) in enumerate(premodel.batch_generator(pair_x, pair_y)):

            y = torch.tensor(y, device= params.device)
            output = premodel(x, y)
            pre_optimizer.zero_grad()
            loss = pre_loss_func(output, y)            
            if i % 50 == 0:
                self.logger.info("MSE loss: %.4f" % loss)
            loss.backward()
            pre_optimizer.step()

        proj_saved_path = os.path.join(self.params.dump_path, "proj.pth")
        torch.save({"projection_matrix" : premodel.state_dict() },proj_saved_path)



    def do_train(self):
        # self.do_pretrain()

        dataloader_tr, dataloader_val, dataloader_test, train_tag2idx, dev_test_tag2idx = get_dataloader(params.tgt_domain, batch_size=params.batch_size, fpath=params.file_path, bert_path=params.bert_path)
        
        premodel = None
        if os.path.exists(os.path.join(self.params.dump_path, "proj.pth")):
            premodel = torch.load(os.path.join(self.params.dump_path, "proj.pth"), map_location=params.device)

             

        dev_test_idx2tag = {v:k for k,v in dev_test_tag2idx.items()}
        self.dev_test_idx2tag = dev_test_idx2tag
        self.train_tag2idx, self.dev_test_tag2idx = train_tag2idx, dev_test_tag2idx
        
        self.slt = Model.SlotFilling(params, train_tag2idx, dev_test_tag2idx, bert_path=params.bert_path,device=params.device)
        if premodel is not None:
            self.slt.Proj_W = premodel.Proj


        self.optimizer = torch.optim.Adam(self.slt.parameters(), lr = params.lr)


        self.slt.to(params.device)
        
        best_dev_f1 = 0
        best_test_f1 = 0
        best_slot_f1 = None
        for epoch in range(params.epoch):
            self.slt.train()
            pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
            loss_list = []
            for i,(words, x, is_heads, pad_heads, tags, y, domains, seq_len) in pbar:
                y = y.to(params.device)
                bsz, max_len = x.size(0), x.size(1)
                x = x.to(params.device)
                pad_heads = pad_heads.to(params.device)
                score = self.slt(x=x, heads=pad_heads, seq_len=seq_len, y=y, domains=domains)
                self.optimizer.zero_grad()
                loss = self.loss_func(score.view(bsz*max_len, -1), y.view(-1))
                loss.backward()
                loss_list.append(loss.item())
                self.optimizer.step()
                pbar.set_description("(Epoch {}) LOSS:{:.4f} ".format((epoch+1), np.mean(loss_list)))

                
            dev_f1, di_dev = self.do_test(dataloader_val)
            test_f1, di_test = self.do_test(dataloader_test)
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1

                best_test_f1 = test_f1
                best_slot_f1 = di_test

                self.save_model()
        self.logger.info("best f1 in test: %.4f" % best_test_f1)
        json_file = os.path.join(self.params.dump_path, "slot_f1.json")
        with open(json_file,'w') as f:
            f.write(json.dumps(di_test) + '\n')
        f.close()

    def do_test(self, data_gen):
        self.slt.eval()
        pbar = tqdm(enumerate(data_gen), total=len(data_gen))
        gold = []
        pred = []
        for i,(words, x, is_heads, pad_heads, tags, y, domains, seq_len) in pbar:
            y = y.to(params.device)
            bsz, max_len = x.size(0), x.size(1)
            x = x.to(params.device)
            pad_heads = pad_heads.to(params.device)
            score = self.slt(x=x, heads=pad_heads, seq_len=seq_len, y=y, iseval = True, domains=domains)
            score = torch.softmax(score, -1)
            score = score.argmax(-1)

            for j in range(len(pad_heads)):
                _pred = []
                _gold = []
                for k in range(1, seq_len[j] - 1):
                    if pad_heads[j][k].item() == 1:
                        _pred.append(self.dev_test_idx2tag[score[j][k].item()])
                        _gold.append(self.dev_test_idx2tag[y[j][k].item()])

                gold.append(_gold)
                pred.append(_pred)
        
        g = []
        p = []
        for i in gold:
            for j in i:
                g.append(j)

        for i in pred:
            for j in i:
                if j == "<PAD>":
                    p.append("O")
                else:
                    p.append(j)
        (prec, rec, f1), di = evaluate(g, p, self.logger)
        return f1, di

    def save_model(self):
        model_saved_path = os.path.join(self.params.dump_path, "best_model.pth")
       
        torch.save({
                "model": self.slt.state_dict(),
                "train_tag2idx":self.train_tag2idx,
                "dev_test_tag2idx":self.dev_test_tag2idx,
            }, model_saved_path)
        logger.info("Best model has been saved to %s" % model_saved_path)

        opti_saved_path = os.path.join(self.params.dump_path, "opti.pth")
        torch.save(self.optimizer.state_dict(), opti_saved_path)
        logger.info("Best model opti has been saved to %s" % opti_saved_path)

        self.model_saved_path = model_saved_path
        self.opti_saved_path = opti_saved_path
          


if __name__ == "__main__":
    params = get_params()
    logger = init_experiment(params, params.logger_filename)
    if params.model_type == "train":
        train_process = Main(params, logger)
        train_process.do_train()


