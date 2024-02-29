DATASET_NAME=kp20k
PROJECT_PATH=/UniKP/UniKeyphrase
DATA_DIR=${PROJECT_PATH}/processed
# Note: output dir should not have any checkpoints if unilm needs to be loaded for finetuning
# if the output dir has any finetuned checkpoints (format *model.{x}.bin) then this checkpoint is used instead of 
# the one specified in the model recovery path
OUTPUT_DIR=${PROJECT_PATH}/finetuned_models/${DATASET_NAME}_5ep_b128_l384_ddp_final_corr1
MODEL_RECOVER_PATH=${PROJECT_PATH}/pretrained_models/unilm/unilm1.2-base-uncased.bin
export CUDA_VISIBLE_DEVICES=0,1,2,3
# for unilm1.2 and 2 remove --new_segment_ids as only 2 token_type_ids exists
# for unilm1 add it as there are 6 token type ids in this case
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 run_seq2seq.py \
  --do_train --use_bwloss --use_SRL --local_rank 0 --num_workers 4 \
  --bert_model bert-base-uncased --do_lower_case \
  --tokenized_input \
  --data_dir ${DATA_DIR} --src_file ${DATASET_NAME}.train.seq.in --tgt_file ${DATASET_NAME}.train.absent \
  --label_file ${DATASET_NAME}.train.seq.out \
  --output_dir ${OUTPUT_DIR} \
  --log_dir ${OUTPUT_DIR} \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 384 --max_position_embeddings 384 \
  --max_len_b 40 \
  --mask_prob 0.7 --max_pred 40 \
  --train_batch_size 128 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 5
