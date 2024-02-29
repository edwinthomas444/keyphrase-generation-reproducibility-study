DATASET_NAME=kp20k
# DATASET_NAME=kptimes
# DATASET_NAME=openkp
# DATASET_NAME=stackexchange

# for Transet base
python train.py \
    --model=TransformerSeq2Set \
    --model_type=seq2set \
    --times=1 \
    --dataset=${DATASET_NAME} \
    --device=cuda:0 \
    --fold_name "${DATASET_NAME}/trans_set_rebuttal_distributed" \
    --example_display_step 50 \

# # kill -9 $(ps -A | grep python | awk '{print $1}')

# for Transet KPDrop(with Replacement)
python train.py \
    --model=TransformerSeq2SetKPD0_7 \
    --model_type=seq2set \
    --times=1 \
    --dataset=${DATASET_NAME} \
    --device=cuda:0 \
    --fold_name "${DATASET_NAME}/trans_set_KPRep_b128_n4_ep5" \
    --example_display_step 50 \

# kill -9 $(ps -A | grep python | awk '{print $1}')

# for Transet KPDrop(with Augmentation)
python train.py \
    --model=TransformerSeq2SetKPD0_7A \
    --model_type=seq2set \
    --times=1 \
    --dataset=${DATASET_NAME} \
    --device=cuda:0 \
    --fold_name "${DATASET_NAME}/trans_set_KPAug_b128_n4_ep5" \
    --example_display_step 50 \

kill -9 $(ps -A | grep python | awk '{print $1}')