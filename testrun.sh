# tgt_domains=(AddToPlaylist BookRestaurant GetWeather PlayMusic RateBook SearchScreeningEvent SearchCreativeWork)
# gamma_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# for tag_dm in ${tgt_domains[@]}
# do
#   for gam in ${gamma_list[@]}
#   do
#     python main.py \
#     --exp_name coach_bert_encoder  \
#     --exp_id ${tag_dm} \
#     --tgt_domain ${tag_dm} \
#     --model_type train \
#     --device cuda:0 \
#     --dump_path /home/sh/data/experiments \
#     --bert_path /home/sh/bert-base-uncased \
#     --file_path /home/sh/data/coachdata/snips \
#     --corps_path /home/sh/data/corps.txt \
#     --emb_file /home/sh/data \
#     --emb_src Bert \
#     --proj no \
#     --gamma ${gam}
#     done
# done

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
--gamma 0.6 \
--log_file /home/sh/data/experiments/coach_bert_encoder \
--proj no