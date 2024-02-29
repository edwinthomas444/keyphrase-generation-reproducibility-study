DATASET_NAME=kp20k
PROJECT_PATH=/UniKP/UniKeyphrase
DATA_DIR=${PROJECT_PATH}/processed
OUTPUT_DIR=${PROJECT_PATH}/finetuned_models/
MODEL_RECOVER_PATH=${PROJECT_PATH}/pretrained_models/unilm/unilm1-base-cased.bin
export CUDA_VISIBLE_DEVICES=0
python run_seq2seq.py \
  --do_train --use_bwloss --use_SRL --num_workers 4 \
  --bert_model bert-base-cased \
  --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} --src_file ${DATASET_NAME}.train.seq.in --tgt_file ${DATASET_NAME}.train.absent \
  --label_file ${DATASET_NAME}.train.seq.out \
  --output_dir ${OUTPUT_DIR} \
  --log_dir ${OUTPUT_DIR} \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 192 --max_position_embeddings 192 \
  --max_len_b 32 \
  --mask_prob 0.7 --max_pred 32 \
  --train_batch_size 64 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 1
