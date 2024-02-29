#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=eval_kp_one2one_kp20k-v1-gpu
#SBATCH --output=slurm_output/eval_kp_one2one_kp20k-v1-gpu.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Run the job
srun python kp_gen_eval.py -config config/test/config-test-keyphrase-one2one.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels/ -output_dir output/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels/ -testsets kp20k -gpu 0 -tasks pred
