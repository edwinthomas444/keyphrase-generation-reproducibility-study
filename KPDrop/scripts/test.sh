
DATASET_NAME=kp20k
# DATASET_NAME=kptimes
# DATASET_NAME=openkp
# DATASET_NAME=stackexchange

decode_modes=("Greedy" "Beam" "TopBeam5LN")
metric_fold_names=("final_results_greedy" "final_results_allbeam" "final_result_top5beam")

# TransSetBase
for i in ${!decode_modes[@]}; do
    python test.py --model=TransformerSeq2Set \
    --model_type=seq2set --times=1 --load_checkpoint \
    --dataset=${DATASET_NAME} --device=cuda:0 --fold_name "${DATASET_NAME}/trans_set_b128_n4_ep5" \
    --decode_mode ${decode_modes[$i]} --metric_fold_name ${metric_fold_names[$i]}
    kill -9 $(ps -A | grep python | awk '{print $1}')
done

# # free gpus
# kill -9 $(ps -A | grep python | awk '{print $1}')

# TransSet+KPDrop-R
for i in ${!decode_modes[@]}; do
    python test.py --model=TransformerSeq2SetKPD0_7 \
    --model_type=seq2set --times=1 --load_checkpoint \
    --dataset=${DATASET_NAME} --device=cuda:0 --fold_name "${DATASET_NAME}/trans_set_KPRep_b128_n4_ep5" \
    --decode_mode ${decode_modes[$i]} --metric_fold_name ${metric_fold_names[$i]}
    kill -9 $(ps -A | grep python | awk '{print $1}')
done

# # free gpus
# kill -9 $(ps -A | grep python | awk '{print $1}')

# TransSet+KPDrop-A
for i in ${!decode_modes[@]}; do
    python test.py --model=TransformerSeq2SetKPD0_7A \
    --model_type=seq2set --times=1 --load_checkpoint \
    --dataset=${DATASET_NAME} --device=cuda:0 --fold_name "${DATASET_NAME}/trans_set_KPAug_b128_n4_ep5" \
    --decode_mode ${decode_modes[$i]} --metric_fold_name ${metric_fold_names[$i]}
    kill -9 $(ps -A | grep python | awk '{print $1}')
done

