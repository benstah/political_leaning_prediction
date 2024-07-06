#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=base_political_leaning
#SBATCH --output=base_political_leaning%j.out
#SBATCH --cpus-per-task=2 # number of cpus
#SBATCH --mem=32G # memory pool for all cores
#SBATCH --gres=gpu:1 # number of gpus

source /storage/sedovaa20/knodle_venv/bin/activate
cd /storage/sedovaa20/benedikt/political_leaning_prediction
python3 src/models/train_model.py