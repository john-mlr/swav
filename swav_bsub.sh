#!/bin/bash

#BSUB -J "mammo swav"
#BSUB -P acc_shenl03_ml
#BSUB -q gpu
#BSUB -R a100
#BSUB -R "rusage[ngpus_excl_p=4]"
#BSUB -R rusage[mem=8000]
#BSUB -n 16
#BSUB -W 72:00
#BSUB -oo /sc/arion/projects/shenl03_ml/2021_john_simCLR/mammo_swav/output.txt
#BSUB -eo /sc/arion/projects/shenl03_ml/2021_john_simCLR/mammo_swav/error.txt

cd /sc/arion/projects/shenl03_ml/2021_john_simCLR/swav

ml purge
ml proxies
ml anaconda3/4.6.4
ml cuda/11.1

source activate Torch_DL

python -m torch.distributed.launch --nproc_per_node=4 main_swav.py \
--data_path /sc/arion/projects/shenl03_ml/2021_john_simCLR/patches/train \
--epochs 100 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 64 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 false \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--epoch_queue_starts 15 \
--dump_path /sc/arion/projects/shenl03_ml/2021_john_simCLR/mammo_swav
