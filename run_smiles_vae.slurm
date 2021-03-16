#!/bin/bash  
#SBATCH -N 1  
#SBATCH -n 16  
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1  
#SBATCH --nodelist=node1237  
#SBATCH --constraint=centos7
#SBATCH --partition=sched_mit_ccoley

source ~/.bashrc
conda activate rdkit
export PYTHONPATH=/home/whgao/test/MolVAE ## path to your MolVAE dir
export CUDA_VISIBLE_DEVICES=0 
python scripts/train.py smiles_vae --device cuda:0 
