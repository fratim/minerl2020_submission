#!/bin/bash
#SBATCH --job-name=assist2                    # Job name
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=frtim@robots.ox.ac.uk    # Where to send mail
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --mem=150gb                          # Job memory request
#SBATCH --time=71:59:00                     # Time limit hrs:min:sec
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:1                        # Requesting 2 GPUs
#SBATCH --output=/scratch/shared/beegfs/frtim/out_assist2.txt
#SBATCH --error=/scratch/shared/beegfs/frtim/error_assist2.txt
# -------------------------------

source /users/frtim/.bashrc

module load cudnn/v8_11.3

conda activate env_subm20

cd /users/frtim/Code/ma_minerl_training

/usr/bin/xvfb-run -s "-ac -screen 0 1280x1024x24" python train.py --env MineRLObtainMASingleVectorObf-v0 --gpu 0

