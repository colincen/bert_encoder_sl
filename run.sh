tgt_domains=(SearchScreeningEvent)
# BookRestaurant GetWeather PlayMusic RateBook SearchCreativeWork SearchScreeningEvent)
exp_n=zero_shot_logits
for tag_dm in ${tgt_domains[@]}
do
``
    python main.py \
    --exp_name ${exp_n}  \
    --exp_id ${tag_dm} \
    --tgt_domain ${tag_dm} \
    --model_type train \
    --device cuda:0 \
    --dump_path /home/shenhao/data/experiments \
    --bert_path /home/shenhao/bert-base-uncased \
    --file_path /home/shenhao/data/coachdata/snips \
    --log_file /home/shenhao/data/experiments/${exp_n} \
    --corps_path /home/shenhao/data/corps.txt \
    --emb_file /home/shenhao/data \
    --model_saved_path /home/shenhao/data/experiments/${exp_n}/${tag_dm} \
    --emb_src Bert \
    --n_samples 0 \
    --coarse_num 7 \
    --gamma  2 \
    --random_select_slot 0 \
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