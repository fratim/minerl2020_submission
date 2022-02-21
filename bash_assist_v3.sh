#!/bin/bash
#SBATCH --job-name=assist_v3_3_seed11                    # Job name
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --mem=40gb                          # Job memory request
#SBATCH --time=167:59:00                     # Time limit hrs:min:sec
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:1                        # Requesting 2 GPUs
#SBATCH --output=/scratch/shared/beegfs/frtim/runs_v2/logs/assist_v3_3_seed11.txt
#SBATCH --error=/scratch/shared/beegfs/frtim/runs_v2/logs/assist_v3_3_seed11.txt
#SBATCH --constraint=rtx6k             # Request nodes with specific features
#SBATCH --exclude=gnodeh2,gnodec1             # Request nodes with specific features
# -------------------------------

source /users/frtim/.bashrc

module load cudnn/v8_11.3

conda activate env_subm20_is3

export GRADLE_USER_HOME=$tri_scratch_dir/runs_v2/gradles/gradles_assist_v3_3_seed11

mkdir $GRADLE_USER_HOME

cd /users/frtim/Code_onlyremote/minerl2020_submission

/usr/bin/xvfb-run -s "-ac -screen 0 1280x1024x24" python train.py --env MineRLObtainMASingleVectorObf-v0 --gpu 0 --outdir $tri_scratch_dir/runs_v2/out/assist_v3_3_seed11 --seed 6

