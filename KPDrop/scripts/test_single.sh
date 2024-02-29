python train.py --model=TransformerSeq2Set \
    --model_type=seq2set --times=1 --load_checkpoint \
    --dataset=kp20k --device=cuda:0 --fold_name trans_set_b128_n4_ep5 \
    --decode_mode "TopBeam5LN" --metric_fold_name "GreedyModelOuts" --example_display_step 50 \
    --limit 60