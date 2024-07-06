#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --job-name=political_leaning
#SBATCH --output=political_leaning%j.out
#SBATCH --cpus-per-task 2 # number of cpus
#SBATCH --mem 64g # memory pool for all cores
#SBATCH -o myoutput.out # STDOUT
#SBATCH -e myerroutput.out # STDERR
#SBATCH --gres=gpu:1 # number of gpus

source /storage/sedovaa20/benedikt/political_leaning_prediction/bin/activate
cd /storage/sedovaa20/benedikt/political_leaning_prediction
python3 src/models/k_fold_model.py