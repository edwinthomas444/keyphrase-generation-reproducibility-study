#!/usr/bin/env bash


source ~/.bash_profile # reload LD_LIBRARY due to error ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
cd /zfs1/hdaqing/rum20/kp/fairseq-kpg

PYTHONUNBUFFERED=1;TOKENIZERS_PARALLELISM=false;CUDA_VISIBLE_DEVICES=0 nohup /ihome/hdaqing/rum20/anaconda3/envs/kp/bin/python3.7 train.py /zfs1/hdaqing/rum20/kp/data/kp/json/kp20k_train1k/train.json --dataset-type scipaper --save-dir /zfs1/hdaqing/rum20/kp/transfer_exps_v2/bart_PTDAFT/bart-PT_step40k-DA_kp20k_step5k-FT1e5_fewshot1k-bs16_step4k_clip01_labelsmooth01/ckpts --disable-validation --task keyphrasification --max-source-length 512 --max-target-length 128 --kp-concat-type pres_abs --arch bart_large --restore-file /zfs1/hdaqing/rum20/kp/transfer_exps_v2/bart_PTDA/bart-PT_wiki_step40k-DA_kp20k_tlrs55-lr1e5-bs256-step5k/ckpts/checkpoint_step_5000.pt --bpe hf_pretrained_bpe --bpe-vocab /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/vocab.json --bpe-merges /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/merges.txt --dict-path /zfs1/hdaqing/rum20/kp/data/kp/hf_vocab/roberta-base-kp/dict.txt --bpe-dropout 0.0 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy --share-all-embeddings --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --optimizer adam --adam-betas (0.9,0.999) --adam-eps 1e-08 --lr 1e-5 --lr-scheduler polynomial_decay --clip-norm 0.1 --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --log-format simple --log-interval 100 --fixed-validation-seed 7 --batch-size 16 --update-freq 1 --save-interval-updates 200 --warmup-updates 400 --max-update 4000 --total-num-update 4000 --num-workers 4 --find-unused-parameters --fp16 --ddp-backend=no_c10d --wandb-project transfer_v2_bart