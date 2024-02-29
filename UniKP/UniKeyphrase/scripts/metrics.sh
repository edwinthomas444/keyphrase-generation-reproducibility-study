DATASET_NAME=kp20k
# DATASET_NAME=kptimes
# DATASET_NAME=openkp
# DATASET_NAME=stackexchange

MODEL_DIRS=("${DATASET_NAME}_5ep_b128_l384_ddp_final_Beam" "${DATASET_NAME}_5ep_b128_l384_ddp_final_BeamNGram" "${DATASET_NAME}_5ep_b128_l384_ddp_final_Greedy")
DATASET_NAMES=(${DATASET_NAME} "nus" "semeval" "krapivin" "inspec")
PROJECT_DIR=/UniKP/UniKeyphrase
OUT_FILE=metrics

# find P,R,F1 for all keyphrases
# python evaluate/evaluate_all.py \
#     ${PROJECT_DIR}/results/${DATASET_NAME}/${MODEL_DIR}/output \
#     ${PROJECT_DIR}/processed/${DATASET_NAME}.test.absent \
#     ${PROJECT_DIR}/results/${DATASET_NAME}/${MODEL_DIR}/output_labels \
#     ${PROJECT_DIR}/processed/${DATASET_NAME}.test.seq.in \
#     ${PROJECT_DIR}/processed/${DATASET_NAME}.test.present \
#     ${PROJECT_DIR}/results/${DATASET_NAME}/${MODEL_DIR}/output_labels.prob \
#     > ${PROJECT_DIR}/results/${DATASET_NAME}/${MODEL_DIR}/${OUT_FILE}

# echo -e '\n' >> ${PROJECT_DIR}/results/${DATASET_NAME}/${MODEL_DIR}/${OUT_FILE}

# find P,R,F1 for present keyphrases

for i in ${!DATASET_NAMES[@]}; do
    for j in ${MODEL_DIRS[@]}; do

        MODEL_DIR=${j}
        echo ${j}
        echo ${DATASET_NAMES[$i]}
        python evaluate/evaluate_present.py \
            --stem_evaluate --stem_divide \
            --label_path ${PROJECT_DIR}/results/${DATASET_NAMES[$i]}/${MODEL_DIR}/output_labels \
            --input_path ${PROJECT_DIR}/processed/${DATASET_NAMES[$i]}.test.seq.in \
            --gold_path ${PROJECT_DIR}/processed/${DATASET_NAMES[$i]}.test.present.test \
            --prob_path ${PROJECT_DIR}/results/${DATASET_NAMES[$i]}/${MODEL_DIR}/output_labels.prob \
            --output_path ${PROJECT_DIR}/results/${DATASET_NAMES[$i]}/${MODEL_DIR} \
            > ${PROJECT_DIR}/results/${DATASET_NAMES[$i]}/${MODEL_DIR}/${OUT_FILE}

        echo -e '\n' >> ${PROJECT_DIR}/results/${DATASET_NAMES[$i]}/${MODEL_DIR}/${OUT_FILE}
        # find P,R,F1 for absent keyphrases
        python evaluate/evaluate_absent.py \
            --stem_evaluate --stem_divide \
            --pred_path ${PROJECT_DIR}/results/${DATASET_NAMES[$i]}/${MODEL_DIR}/output \
            --input_path ${PROJECT_DIR}/processed/${DATASET_NAMES[$i]}.test.seq.in \
            --gold_path ${PROJECT_DIR}/processed/${DATASET_NAMES[$i]}.test.absent \
            --output_path ${PROJECT_DIR}/results/${DATASET_NAMES[$i]}/${MODEL_DIR} \
            >> ${PROJECT_DIR}/results/${DATASET_NAMES[$i]}/${MODEL_DIR}/${OUT_FILE}
    done
done

