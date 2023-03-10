#!/bin/bash -lT


#SBATCH --job-name=EDL_RES_1
#SBATCH --time 00-15:20:00
#SBATCH --account cisc-896 --partition tier3
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH --gres=gpu:v100:1

conda activate sn_coreset

python main_job.py -b $ss_size -c 350 -bs 256 -e 50  