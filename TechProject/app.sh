#!/bin/bash

#SBATCH -o app.o # 把输出结果STDOUT保存在哪一个文件
#SBATCH -e app.e # 把报错结果STDERR保存在哪一个文件
#SBATCH --nodelist=gpu05 # 需要使用的节点
#SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
#SBATCH --time=100-00:00:00
#SBATCH --mem=100G
#SBATCH --job-name=tech
#SBATCH -c2

flask run --host=0.0.0.0
