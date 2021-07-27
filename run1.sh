tgt_domains=(AddToPlaylist SearchCreativeWork GetWeather)
for tag_dm in ${tgt_domains[@]}
do

    python main.py \
    --exp_name zero_shot_logits  \
    --exp_id ${tag_dm} \
    --tgt_domain ${tag_dm} \
    --model_type train \
    --device cuda:1 \
    --dump_path /home/shenhao/data/experiments \
    --bert_path /home/shenhao/bert-base-uncased \
    --file_path /home/shenhao/data/coachdata/snips \
    --log_file /home/shenhao/data/experiments/zero_shot_logits \
    --corps_path /home/shenhao/data/corps.txt \
    --emb_file /home/shenhao/data \
    --model_saved_path /home/shenhao/data/experiments/zero_shot_logits/${tag_dm} \
    --emb_src Bert \
    --n_samples 0 \
    --coarse_num 5 \
    --gamma  2 \
    --proj no
done
