import src.model as Model
from src.model import ProjMartrix
import torch
import torch.nn as nn
from src.datareader import get_dataloader, coarse, bins_labels,father_son_slot
from src.conlleval import evaluate
from config import get_params, init_experiment
from tqdm import tqdm
import numpy as np
import os
import json
import functools
from prettytable import PrettyTable
import csv
from src.utils import load_embedding

class Main:
    def __init__(self, params, logger):
        self.params = params
        self.logger = logger
        self.loss_func = nn.CrossEntropyLoss()
        self.mse_loss_func = nn.MSELoss()
        self.model_saved_path = None
        self.opti_saved_path = None

    def do_train(self):
        dataloader_tr, dataloader_val, dataloader_test, vocab, train_tag2idx, dev_test_tag2idx, train_mask, test_mask \
            = get_dataloader(params.tgt_domain, params.batch_size, params.n_samples)

        
        embeddings = load_embedding(vocab, 300, '/home/sh/data/glove.6B.300d.txt')

        dev_test_idx2tag = {v:k for k,v in dev_test_tag2idx.items()}
        self.dev_test_idx2tag = dev_test_idx2tag
        self.train_tag2idx, self.dev_test_tag2idx = train_tag2idx, dev_test_tag2idx
        

        self.slt = Model.SlotFilling(params, embeddings,  train_tag2idx, dev_test_tag2idx,device=params.device)

        self.optimizer = torch.optim.Adam(self.slt.parameters(), lr = params.lr)


        self.slt.to(params.device)
        


        best_dev_f1 = 0
        best_test_f1 = 0
        best_slot_f1 = None
        patience = 0
        for epoch in range(params.epoch):
            self.slt.train()
            pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
            
            total_loss_list = []
            coarse_loss_list = []
            sim_loss_list = []
            mse_loss_list = []

            for i, (x, seq_len,  y0, y1, domains) in pbar:
                y1 = y1.to(params.device)
                y0 = y0.to(params.device)
                seq_len = seq_len.to(params.device)
                bsz, max_len = x.size(0), x.size(1)
                x = x.to(params.device)
                coarse_logits, final_logits, loss_coarse, loss_sim = self.slt(x=x,  seq_len=seq_len, y0=y0, y1=y1, logits_mask = train_mask, alpha = epoch)

                self.optimizer.zero_grad()

                if epoch < 2:
                    loss = loss_coarse 
                else:
                    loss = loss_sim  + loss_coarse
                # loss = self.params.gamma * (loss_sim + mseloss) + (1 - self.params.gamma) * loss_coarse

                loss.backward()
                total_loss_list.append(loss.item())
                coarse_loss_list.append(loss_coarse.item())
                sim_loss_list.append(loss_sim.item())
                # mse_loss_list.append(mseloss.item())

                self.optimizer.step()
                pbar.set_description("(Epoch {}) Coarse LOSS:{:.4f}  Sim LOSS:{:.4f} Total Loss{:.4f}".format \
                ((epoch+1), \
                    np.mean(coarse_loss_list), \
                    # np.mean(mse_loss_list), \
                    np.mean(sim_loss_list), \
                    np.mean(total_loss_list)
                    ))
                if i > 0 and i % 50 == 0:
                    # continue
                    dev_f1, di_dev = self.do_test(dataloader_val, test_mask)
                    test_f1, di_test = self.do_test(dataloader_test, test_mask)
                    if dev_f1 > best_dev_f1:
                        patience = 0
                        best_dev_f1 = dev_f1

                        best_test_f1 = test_f1
                        best_slot_f1 = di_test
                        self.save_model()                    
                
            dev_f1, di_dev = self.do_test(dataloader_val, test_mask)
            test_f1, di_test = self.do_test(dataloader_test, test_mask)
            if dev_f1 > best_dev_f1:
                patience = 0
                best_dev_f1 = dev_f1

                best_test_f1 = test_f1
                best_slot_f1 = di_test
                self.save_model()
            else:
                patience += 1
                if patience > 5:
                    break

                
        self.logger.info("best f1 in test: %.4f" % best_test_f1)
        json_file = os.path.join(self.params.log_file, str(self.params.exp_id) + "_" + str(self.params.gamma) + ".json")
        with open(json_file,'w') as f:
            f.write(json.dumps(best_slot_f1) + '\n')
        f.close()
        

    def finetag_to_coarsetag(self):
        res_dict = {'<PAD>':"pad", "O":"O"}
        for k,v in father_son_slot.items():
            # res_dict["B-" + k] = len(res_dict)
            # res_dict["I-" + k] = len(res_dict)
            for t in v:
                _B = "B-" + t
                if _B not in res_dict.keys():
                    res_dict[_B] = "B-" + k
                _I = "I-" + t
                if _I not in res_dict.keys():
                    res_dict[_I] = "I-" + k 
        return res_dict
   
    def do_test(self, data_gen, test_mask, badcase=True):

        self.slt.eval()
        pbar = tqdm(enumerate(data_gen), total=len(data_gen))
        gold = []
        pred = []


        fine2coarsetag = self.finetag_to_coarsetag()
        coarse_gold = []
        coarse_pred = []
        word_list = []
        for i, (x, seq_len,  y0, y1, domains) in pbar:
            y1 = y1.to(params.device)
            y0 = y0.to(params.device)
            seq_len = seq_len.to(params.device)
            bsz, max_len = x.size(0), x.size(1)
            x = x.to(params.device)
            coarse_logits, final_logits, loss_coarse, loss_sim = self.slt(x=x,  seq_len=seq_len, y0=y0, y1=y1, logits_mask = test_mask, alpha = 0, iseval=True)
            # score = torch.softmax(final_logits, -1)
            # score = score.argmax(-1)

            #----------------
            attention_mask = (x != 0).byte().to(params.device)

            coarse_score, best_list = self.slt.crf.decode(coarse_logits, attention_mask)
            emb_list  = self.slt.crf_labemb.decode(final_logits, attention_mask)
            #------------------
            # print(coarse_score.size())
            # coarse_score = coarse_score.argmax(-1)
            # print(coarse_score.size())
            # coarse_score = torch.softmax(coarse_logits, -1)
            # emb_list = final_logits.argmax(-1)

            for j in range(len(x)):
                _pred = []
                _gold = []

                _coarse_pred = []
                _coarse_gold = []

                for k in range(1, seq_len[j] - 1):
                    pred_fine_tag = self.dev_test_idx2tag[emb_list[j][k]]
                    
                    gold_fine_tag = self.dev_test_idx2tag[y1[j][k].item()]

                    _pred.append(pred_fine_tag)
                    _gold.append(gold_fine_tag)


                    if bins_labels[y0[j][k].item()] == 'pad':
                        _coarse_gold.append('O')
                    else:      
                        _coarse_gold.append(bins_labels[y0[j][k].item()])



                    if bins_labels[best_list[j][k]] == 'pad':
                        _coarse_pred.append('O')
                    else:      
                        _coarse_pred.append(bins_labels[best_list[j][k]])




                gold.append(_gold)
                pred.append(_pred)
                coarse_gold.append(_coarse_gold)
                coarse_pred.append(_coarse_pred)
        
        self.slt.train()
        badcase = False
        if badcase:



            json_file = os.path.join(self.params.log_file, str(self.params.exp_id) + "_" + str(self.params.gamma) + "_bad_case.txt")


            with open(json_file,'w') as f:
               for i in range(len(gold)):
                   for j in range(len(gold[i])):
                       if gold[i][j] != pred[i][j]:
                           f.write(str(word_list[i].split(' ')[1:-1])+'\n')
                           f.write('wrong:  ' + str(pred[i])+'\n')
                           f.write('right:  ' + str(gold[i])+'\n')
                           f.write('\n\n')
                           break
            f.close()

        # print(gold)
        # print(pred)
        # print(word_list)

        
        
        g = []
        p = []
        for i in coarse_gold:
            for j in i:
                g.append(j)

        for i in coarse_pred:
            for j in i:
                if j == "<PAD>" or j == 'pad':
                    p.append("O")
                else:
                    p.append(j)
        (prec, rec, f1), di = evaluate(g, p, self.logger)


        g.clear()
        p.clear()
        for i in gold:
            for j in i:
                g.append(j)

        for i in pred:
            for j in i:
                if j == "<PAD>":
                    p.append("O")
                else:
                    p.append(j)






        slot_set = set()
        for i in g:
            slot_set.add(i)
        for i in p:
            slot_set.add(i)    
        slot_l = list(slot_set)
        slot_l = sorted(slot_l, key=lambda x : x[2:])
        


        matrix_dict = {}
        
        for i in slot_l:
            if i not in matrix_dict:
                matrix_dict[i] = len(matrix_dict)
        

        id2slot = {v : k for k,v in matrix_dict.items()}


        cnt = [[0 for col in range(len(matrix_dict))] for row in range(len(matrix_dict))]


        temp_keys = [id2slot[j] for j in range(len(id2slot))]
        tb = PrettyTable()
        tb.field_names = ["a"] + temp_keys
        csv_row = ["a"] + temp_keys
        csv_file = os.path.join(self.params.log_file, str(self.params.exp_id) + "_" + str(self.params.gamma) + "_matrix.csv")
        out = open(csv_file,"w", newline="")
        csv_writer = csv.writer(out,dialect='excel')
        for i in range(len(g)):
            row_idx = matrix_dict[g[i]]
            col_idx = matrix_dict[p[i]]
            cnt[row_idx][col_idx] += 1


        csv_writer.writerow(csv_row)
     

        for i in range(len(id2slot)):
            tb.add_row([id2slot[i]] + cnt[i])
            csv_writer.writerow([id2slot[i]] + cnt[i])
        
        
        if badcase:
        
            json_file = os.path.join(self.params.log_file, str(self.params.exp_id) + "_" + str(self.params.gamma) + "_matrix.txt")

            with open(json_file,'w') as f:
                f.write(str(tb))
            f.close()        




        
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

    def do_test_prog(self):
        model_saved_path = self.params.model_saved_path
        best_model = os.path.join(model_saved_path,'best_model.pth')
        res_dict = torch.load(best_model)
        model_params = res_dict['model']
        train_tag2idx = res_dict['train_tag2idx']
        dev_test_tag2idx = res_dict['dev_test_tag2idx']

        self.slt = Model.SlotFilling(params, train_tag2idx, dev_test_tag2idx, bert_path=params.bert_path,device=params.device)
        self.slt.to(params.device)
        self.slt.load_state_dict(model_params)
        dataloader_tr, dataloader_val, dataloader_test, train_tag2idx, dev_test_tag2idx, train_mask, test_mask = get_dataloader(params.tgt_domain, batch_size=params.batch_size, fpath=params.file_path, bert_path=params.bert_path, n_samples=params.n_samples)
        dev_test_idx2tag = {v:k for k,v in dev_test_tag2idx.items()}
        self.dev_test_idx2tag = dev_test_idx2tag
        self.train_tag2idx, self.dev_test_tag2idx = train_tag2idx, dev_test_tag2idx
        self.do_test(dataloader_test, test_mask, True)

if __name__ == "__main__":
    params = get_params()
    logger = init_experiment(params, params.tgt_domain+"_"+ params.logger_filename)
    if params.model_type == "train":
        train_process = Main(params, logger)
        train_process.do_train()
    else:
        test_process = Main(params, logger)
        test_process.do_test_prog()

