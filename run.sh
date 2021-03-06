python main.py \
--exp_name coach_bert_encoder  \
--exp_id atp \
--tgt_domain AddToPlaylist \
--model_type train \
--device cuda:0 \
--dump_path /home/sh/data/experiments \
--bert_path /home/sh/bert-base-uncased \
--file_path /home/sh/data/coachdata/snips \
--corps_path /home/sh/data/corps.txt \
--emb_file /home/sh/data \
--emb_src Bert \
--proj yes

python main.py \
--exp_name coach_bert_encoder  \
--exp_id br \
--tgt_domain BookRestaurant \
--model_type train \
--device cuda:0 \
--dump_path /home/sh/data/experiments \
--bert_path /home/sh/bert-base-uncased \
--file_path /home/sh/data/coachdata/snips \
--corps_path /home/sh/data/corps.txt \
--emb_file /home/sh/data \
--emb_src Bert \
--proj yes

python main.py \
--exp_name coach_bert_encoder  \
--exp_id gw \
--tgt_domain GetWeather \
--model_type train \
--device cuda:0 \
--dump_path /home/sh/data/experiments \
--bert_path /home/sh/bert-base-uncased \
--file_path /home/sh/data/coachdata/snips \
--corps_path /home/sh/data/corps.txt \
--emb_file /home/sh/data \
--emb_src Bert \
--proj yes

python main.py \
--exp_name coach_bert_encoder  \
--exp_id pm \
--tgt_domain PlayMusic \
--model_type train \
--device cuda:0 \
--dump_path /home/sh/data/experiments \
--bert_path /home/sh/bert-base-uncased \
--file_path /home/sh/data/coachdata/snips \
--corps_path /home/sh/data/corps.txt \
--emb_file /home/sh/data \
--emb_src Bert \
--proj yes

python main.py \
--exp_name coach_bert_encoder  \
--exp_id rb \
--tgt_domain RateBook \
--model_type train \
--device cuda:0 \
--dump_path /home/sh/data/experiments \
--bert_path /home/sh/bert-base-uncased \
--file_path /home/sh/data/coachdata/snips \
--corps_path /home/sh/data/corps.txt \
--emb_file /home/sh/data \
--emb_src Bert \
--proj yes

python main.py \
--exp_name coach_bert_encoder  \
--exp_id sse \
--tgt_domain SearchScreeningEvent \
--model_type train \
--device cuda:0 \
--dump_path /home/sh/data/experiments \
--bert_path /home/sh/bert-base-uncased \
--file_path /home/sh/data/coachdata/snips \
--corps_path /home/sh/data/corps.txt \
--emb_file /home/sh/data \
--emb_src Bert \
--proj yes

python main.py \
--exp_name coach_bert_encoder  \
--exp_id sc \
--tgt_domain SearchCreativeWork \
--model_type train \
--device cuda:0 \
--dump_path /home/sh/data/experiments \
--bert_path /home/sh/bert-base-uncased \
--file_path /home/sh/data/coachdata/snips \
--corps_path /home/sh/data/corps.txt \
--emb_file /home/sh/data \
--emb_src Bert \
--proj yes
