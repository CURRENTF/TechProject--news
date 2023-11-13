#!/bin/bash

#SBATCH --partition=si
#SBATCH --gres=gpu:1
#SBATCH -c4
#SBATCH --time=999:00:00
#SBATCH --output=./debug/killer.out
#SBATCH --nodelist=gpu05
#SBATCH --mem=30G

python scripts/inference/img_killer.py