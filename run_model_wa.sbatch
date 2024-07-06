#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=political_leaning
#SBATCH --output=political_leaning_wa%j.out
#SBATCH --cpus-per-task=4 # number of cpus
#SBATCH --mem=120G # memory pool for all cores
#SBATCH --gres=gpu:1 # number of gpus

source /storage/sedovaa20/knodle_venv/bin/activate
cd /storage/sedovaa20/benedikt/political_leaning_prediction
python3 src/models/train_model_with_wa.py