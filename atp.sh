tgt_domains=(AddToPlaylist)
# gamma_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for tag_dm in ${tgt_domains[@]}
do

    python main.py \
    --exp_name zero_shot_logits  \
    --exp_id ${tag_dm} \
    --tgt_domain ${tag_dm} \
    --model_type train \
    --device cuda:0 \
    --dump_path /home/sh/data/experiments \
    --bert_path /home/sh/bert-base-uncased \
    --lr 0.001 \
    --file_path /home/sh/data/coachdata/snips \
    --log_file /home/sh/data/experiments/zero_shot_logits \
    --corps_path /home/sh/data/corps.txt \
    --emb_file /home/sh/data \
    --model_saved_path /home/sh/data/experiments/zero_shot_logits/${tag_dm} \
    --n_samples 0 \
    --proj no

done






# for tag_dm in ${tgt_domains[@]}
# do

#     python main.py \
#     --exp_name few_shot_20  \
#     --exp_id ${tag_dm} \
#     --tgt_domain ${tag_dm} \
#     --model_type train \
#     --device cuda:0 \
#     --dump_path /home/sh/data/experiments \
#     --bert_path /home/sh/bert-base-uncased \
#     --file_path /home/sh/data/coachdata/snips \
#     --log_file /home/sh/data/experiments/few_shot_20 \
#     --corps_path /home/sh/data/corps.txt \
#     --emb_file /home/sh/data \
#     --emb_src Bert \
#     --n_samples 20 \
#     --proj no

# done


# for tag_dm in ${tgt_domains[@]}
# do

#     python main.py \
#     --exp_name few_shot_50  \
#     --exp_id ${tag_dm} \
#     --tgt_domain ${tag_dm} \
#     --model_type train \
#     --device cuda:0 \
#     --dump_path /home/sh/data/experiments \
#     --bert_path /home/sh/bert-base-uncased \
#     --file_path /home/sh/data/coachdata/snips \
#     --log_file /home/sh/data/experiments/few_shot_50 \
#     --corps_path /home/sh/data/corps.txt \
#     --emb_file /home/sh/data \
#     --emb_src Bert \
#     --n_samples 50 \
#     --proj no

# done
