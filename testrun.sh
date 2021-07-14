tgt_domains=(attraction restaurant)
# gamma_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for tag_dm in ${tgt_domains[@]}
do

    python main.py \
    --exp_name woz_data_bert_reps_5  \
    --exp_id ${tag_dm} \
    --tgt_domain ${tag_dm} \
    --model_type train \
    --device cuda:0 \
    --dump_path /home/shenhao/data/experiments \
    --bert_path /home/shenhao/bert-base-uncased \
    --file_path /home/shenhao/data/coachdata/multiwoz \
    --log_file /home/shenhao/data/experiments/woz_data_bert_reps_5 \
    --corps_path /home/shenhao/data/corps.txt \
    --emb_file /home/shenhao/data \
    --model_saved_path /home/shenhao/data/experiments/woz_data_bert_reps_5/${tag_dm} \
    --emb_src Bert \
    --n_samples 0 \
    --proj no

done