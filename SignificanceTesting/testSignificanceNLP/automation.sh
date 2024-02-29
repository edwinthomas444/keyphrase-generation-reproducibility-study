METRIC_TYPES=('M' '5' 'O')
KP_TYPES=('present' 'absent')
TEST_TYPES=('Permutation' 'Bootstrap')
OUTPUT_FILE='output.txt'

# overwrite existing file then append results
echo 'Start of Test' > ${OUTPUT_FILE}

for i in ${!KP_TYPES[@]}; do
    for j in ${!METRIC_TYPES[@]}; do
        for k in ${!TEST_TYPES[@]}; do
            python testSignificance.py "resA_${KP_TYPES[$i]}_${METRIC_TYPES[$j]}.txt" "resB_${KP_TYPES[$i]}_${METRIC_TYPES[$j]}.txt" "0.05" ${TEST_TYPES[$k]} >> ${OUTPUT_FILE}
        done
    done
done
