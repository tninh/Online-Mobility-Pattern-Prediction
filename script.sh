#!/bin/bash
# 
#SBATCH --job-name=295_ap_name
#SBATCH --output=295_ap_name.log
# 
#SBATCH --gres=gpu:1
#SBATCH --mem=4000M
#SBATCH --time=5:00:00

echo "======START======="
module load python3/3.5.6 
#python3 /home/011816337/295/feature_extraction-ap_name-Copy1.py
nvidia-smi
echo "==================end==================="


