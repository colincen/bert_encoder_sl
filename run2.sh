tgt_domains=(SearchCreativeWork)
exp_n=cluster1
for tag_dm in ${tgt_domains[@]}
do

    python main.py \
    --exp_name ${exp_n}  \
    --exp_id ${tag_dm} \
    --tgt_domain ${tag_dm} \
    --model_type train \
    --device cuda:1 \
    --dump_path /home/shenhao/data/experiments \
    --bert_path /home/shenhao/bert-base-uncased \
    --file_path /home/shenhao/data/coachdata/snips \
    --log_file /home/shenhao/data/experiments/${exp_n} \
    --corps_path /home/shenhao/data/corps.txt \
    --emb_file /home/shenhao/data \
    --model_saved_path /home/shenhao/data/experiments/${exp_n}/${tag_dm} \
    --emb_src Bert \
    --n_samples 0 \
    --coarse_num 6 \
    --gamma  2 \
    --random_select_slot 0 \
    --proj no
done

exp_n=cluster2
for tag_dm in ${tgt_domains[@]}
do

    python main.py \
    --exp_name ${exp_n}  \
    --exp_id ${tag_dm} \
    --tgt_domain ${tag_dm} \
    --model_type train \
    --device cuda:1 \
    --dump_path /home/shenhao/data/experiments \
    --bert_path /home/shenhao/bert-base-uncased \
    --file_path /home/shenhao/data/coachdata/snips \
    --log_file /home/shenhao/data/experiments/${exp_n} \
    --corps_path /home/shenhao/data/corps.txt \
    --emb_file /home/shenhao/data \
    --model_saved_path /home/shenhao/data/experiments/${exp_n}/${tag_dm} \
    --emb_src Bert \
    --n_samples 0 \
    --coarse_num 6 \
    --gamma  2 \
    --random_select_slot 0 \
    --proj no
done


exp_n=cluster3
for tag_dm in ${tgt_domains[@]}
do

    python main.py \
    --exp_name ${exp_n}  \
    --exp_id ${tag_dm} \
    --tgt_domain ${tag_dm} \
    --model_type train \
    --device cuda:1 \
    --dump_path /home/shenhao/data/experiments \
    --bert_path /home/shenhao/bert-base-uncased \
    --file_path /home/shenhao/data/coachdata/snips \
    --log_file /home/shenhao/data/experiments/${exp_n} \
    --corps_path /home/shenhao/data/corps.txt \
    --emb_file /home/shenhao/data \
    --model_saved_path /home/shenhao/data/experiments/${exp_n}/${tag_dm} \
    --emb_src Bert \
    --n_samples 0 \
    --coarse_num 6 \
    --gamma  2 \
    --random_select_slot 0 \
    --proj no
done


exp_n=cluster4
for tag_dm in ${tgt_domains[@]}
do

    python main.py \
    --exp_name ${exp_n}  \
    --exp_id ${tag_dm} \
    --tgt_domain ${tag_dm} \
    --model_type train \
    --device cuda:1 \
    --dump_path /home/shenhao/data/experiments \
    --bert_path /home/shenhao/bert-base-uncased \
    --file_path /home/shenhao/data/coachdata/snips \
    --log_file /home/shenhao/data/experiments/${exp_n} \
    --corps_path /home/shenhao/data/corps.txt \
    --emb_file /home/shenhao/data \
    --model_saved_path /home/shenhao/data/experiments/${exp_n}/${tag_dm} \
    --emb_src Bert \
    --n_samples 0 \
    --coarse_num 6 \
    --gamma  2 \
    --random_select_slot 0 \
    --proj no
done
