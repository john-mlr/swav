#!/bin/bash

#BSUB -J "eval swav"
#BSUB -P acc_shenl03_ml
#BSUB -q gpu
#BSUB -R a100
#BSUB -n 8
#BSUB -W 04:00
#BSUB -oo /sc/arion/projects/shenl03_ml/2021_john_simCLR/mammo_swav/trial_2/eval175_output.txt
#BSUB -eo /sc/arion/projects/shenl03_ml/2021_john_simCLR/mammo_swav/trial_2/eval175_error.txt

cd /sc/arion/projects/shenl03_ml/2021_john_simCLR/swav

ml purge
ml proxies
ml anaconda3/4.6.4
ml cuda/11.1

source activate Torch_DL

python linear_eval.py -e 2000