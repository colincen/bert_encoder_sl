import argparse
import os
import subprocess
import pickle
import logging
import time
import random
from datetime import timedelta
import numpy as np
from tqdm import tqdm
import pickle
import numpy as np


def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="Cross-domain SLU")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="cross-domain-slu.log")
    parser.add_argument("--dump_path", type=str, default="/data/sh/experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")

    # adaptation parameters
    parser.add_argument("--epoch", type=int, default=20, help="number of maximum epoch")
    parser.add_argument("--tgt_domain", type=str, default="", help="target_domain")
    parser.add_argument("--bert_path", type=str, default="/data/sh/bert-base-uncased", help="embeddings file")  
    # slu_word_char_embs_with_slotembs.npy
    parser.add_argument("--file_path", type=str, default="/data/sh/coachdata/snips", help="embedding dimension") #400
    parser.add_argument("--corps_path", type=str, default="/home/sh/data/corps.txt", help="corps file") 
    parser.add_argument("--emb_file", type=str, default="/home/sh/data", help="emb file") 
    parser.add_argument("--proj", type=str, default="no", help="emb file")

    parser.add_argument("--emb_dim", type=int, default=768, help="embedding dimension") #400
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--emb_src", type=str, default='Bert', help='embedding source')


    # parser.add_argument("--num_binslot", type=int, default=15, help="number of father slot")
    # parser.add_argument("--num_slot", type=int, default=72, help="number of slot types")
    # parser.add_argument("--num_domain", type=int, default=7, help="number of domain")
    # parser.add_argument("--freeze_emb", default=False, action="store_true", help="Freeze embeddings")
    # parser.add_argument("--pretrained_epoch", type=int, default=3, help="chunking_pretrained")

    # parser.add_argument("--slot_emb_file", type=str, default="/home/sh/data/coachdata/snips/emb/slot_word_char_embs_based_on_each_domain.dict", help="dictionary type: slot embeddings based on each domain") # slot_embs_based_on_each_domain.dict w/o char embeddings  slot_word_char_embs_based_on_each_domain.dict w/ char embeddings
    # parser.add_argument("--visualization_path", type=str, default="/home/sh/data/experiments/vis/")
    # parser.add_argument("--bidirection", default=False, action="store_true", help="Bidirectional lstm")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    # parser.add_argument("--lr_decay", type=float, default=0.5, help="learning rate decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")
    # parser.add_argument("--hidden_dim", type=int, default=200, help="hidden dimension for LSTM")#200
    # parser.add_argument("--n_layer", type=int, default=2, help="number of layers for LSTM")
    # parser.add_argument("--early_stop", type=int, default=5, help="No improvement after several epoch, we stop training")
    # parser.add_argument("--patience", type=int, default=2, help="trivial")
    # parser.add_argument("--binary", default=False, action="store_true", help="conduct binary training only")
    # parser.add_argument("--opti_path", type=str, default="/data/sh/experiments/", help="dictionary type: slot embeddings based on each domain")
    parser.add_argument("--domain", type=str, default="atp", help="dictionary type: slot embeddings based on each domain")
    parser.add_argument("--device", type=str, default="cuda:0")


    # # add label_encoder
    # parser.add_argument("--tr", default=False, action="store_true", help="use template regularization")

    # # few shot learning
    # parser.add_argument("--n_samples", type=int, default=0, help="number of samples for few shot learning")

    # # encoder type for encoding entity tokens in the Step Two
    # parser.add_argument("--enc_type", type=str, default="lstm", help="encoder type for encoding entity tokens (e.g., trs, lstm, none)")

    # # transformer parameters
    # parser.add_argument("--num_heads", type=int, default=4, help="Number of heads for transformer")
    # parser.add_argument("--trs_hidden_dim", type=int, default=400, help="Dimension after combined into word level")#400
    # parser.add_argument("--filter_size", type=int, default=64, help="Hidden size of the middle layer in FFN")
    # parser.add_argument("--dim_key", type=int, default=0, help="Key dimension in transformer (if 0, then would be the same as hidden_size)")
    # parser.add_argument("--dim_value", type=int, default=0, help="Value dimension in transformer (if 0, then would be the same as hidden_size)")
    # parser.add_argument("--trs_layers", type=int, default=1, help="Number of layers for transformer")

    # # baseline
    # parser.add_argument("--use_example", default=False, action="store_true", help="use example value")
    # parser.add_argument("--example_emb_file", type=str, default="/home/sh/data/coachdata/snips/emb/example_embs_based_on_each_domain.dict", help="dictionary type: example embeddings based on each domain")

    # # test model
    # parser.add_argument("--model_path", type=str, default="", help="Saved model path")
    parser.add_argument("--model_type", type=str, default="", help="Saved model type (e.g., coach, ct, rzt)")
    # parser.add_argument("--test_mode", type=str, default="testset", help="Choose mode to test the model (e.g., testset, seen_unseen)")

    # # NER
    # parser.add_argument("--ner_entity_type_emb_file", type=str, default="/home/sh/data/coachdata/ner/emb/entity_type_embs.npy", help="entity type embeddings file path")
    # parser.add_argument("--ner_example_emb_file", type=str, default="/home/sh/data/coachdata/ner/emb/example_embs.npy", help="entity example embeddings file path")
    # parser.add_argument("--bilstmcrf", default=False, action="store_true", help="use BiLSTM-CRF baseline")
    # parser.add_argument("--num_entity_label", type=int, default=9, help="number of entity label")

    params = parser.parse_args()

    return params


def init_experiment(params, logger_filename):
    """
    Initialize the experiment:
    - save parameters
    - create a logger
    """
    # save parameters
    get_saved_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)

    return logger

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def create_logger(filepath):
    # create log formatter
    log_formatter = LogFormatter()
    
    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def get_saved_path(params):
    """
    create a directory to store the experiment
    """
    dump_path = "./" if params.dump_path == "" else params.dump_path
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    assert os.path.isdir(dump_path)

    # create experiment path if it does not exist
    exp_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(exp_path):
        subprocess.Popen("mkdir -p %s" % exp_path, shell=True).wait()
    
    # generate id for this experiment
    if params.exp_id == "":
        chars = "0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(0, 3))
            if not os.path.isdir(os.path.join(exp_path, exp_id)):
                break
    else:
        exp_id = params.exp_id
    # update dump_path
    params.dump_path = os.path.join(exp_path, exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()
    assert os.path.isdir(params.dump_path)