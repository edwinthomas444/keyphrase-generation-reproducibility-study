TRAIN_DATASET_NAME=kp20k
# TRAIN_DATASET_NAME=kptimes
# TRAIN_DATASET_NAME=openkp
# TRAIN_DATASET_NAME=stackexchange

DATASET_NAMES=(${TRAIN_DATASET_NAME} "krapivin" "inspec" "semeval" "nus")
PROJECT_PATH=/UniKP/UniKeyphrase
DATA_DIR=${PROJECT_PATH}/processed
MODEL_RECOVER_PATH=${PROJECT_PATH}/finetuned_models/${TRAIN_DATASET_NAME}_5ep_b128_l384_ddp_final_corr1/model.5.bin
export CUDA_VISIBLE_DEVICES=0


for i in ${!DATASET_NAMES[@]}; do

  RESULT_DIR=${PROJECT_PATH}/results/${DATASET_NAMES[$i]}/${TRAIN_DATASET_NAME}_5ep_b128_l384_ddp_final_Greedy
  EVAL_SPLIT=${DATASET_NAMES[$i]}.test
  echo $RESULT_DIR
  echo $EVAL_SPLIT
  # greedy search
  python decode_seq2seq.py \
    --bert_model bert-base-uncased --do_lower_case --output_probs \
    --new_segment_ids --mode s2s \
    --input_file ${DATA_DIR}/${DATASET_NAMES[$i]}.test.seq.in --split ${EVAL_SPLIT} --tokenized_input \
    --output_file ${RESULT_DIR}/output \
    --output_label_file ${RESULT_DIR}/output_labels \
    --model_recover_path ${MODEL_RECOVER_PATH} \
    --max_seq_length 384 --max_tgt_length 40 \
    --batch_size 64 --beam_size 1 --length_penalty 0 \
    --forbid_ignore_word "."\

  # beam search
  RESULT_DIR=${PROJECT_PATH}/results/${DATASET_NAMES[$i]}/${TRAIN_DATASET_NAME}_5ep_b128_l384_ddp_final_Beam
  python decode_seq2seq.py \
    --bert_model bert-base-uncased --do_lower_case --need_score_traces --output_probs \
    --new_segment_ids --mode s2s \
    --input_file ${DATA_DIR}/${DATASET_NAMES[$i]}.test.seq.in --split ${EVAL_SPLIT} --tokenized_input \
    --output_file ${RESULT_DIR}/output \
    --output_label_file ${RESULT_DIR}/output_labels \
    --model_recover_path ${MODEL_RECOVER_PATH} \
    --max_seq_length 384 --max_tgt_length 40 \
    --batch_size 64 --beam_size 5 --length_penalty 0 \
    --forbid_ignore_word "."
  
  # beam search by forbidding duplicate ngrams
  RESULT_DIR=${PROJECT_PATH}/results/${DATASET_NAMES[$i]}/${TRAIN_DATASET_NAME}_5ep_b128_l384_ddp_final_BeamNGram
  python decode_seq2seq.py \
    --bert_model bert-base-uncased --do_lower_case --need_score_traces --output_probs \
    --new_segment_ids --mode s2s \
    --input_file ${DATA_DIR}/${DATASET_NAMES[$i]}.test.seq.in --split ${EVAL_SPLIT} --tokenized_input \
    --output_file ${RESULT_DIR}/output \
    --output_label_file ${RESULT_DIR}/output_labels \
    --model_recover_path ${MODEL_RECOVER_PATH} \
    --max_seq_length 384 --max_tgt_length 40 \
    --batch_size 64 --beam_size 5 --length_penalty 0 \
    --forbid_duplicate_ngrams --forbid_ignore_word "."

  done
