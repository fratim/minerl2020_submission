#!/bin/bash
#SBATCH --job-name=org_obf_v4_8                    # Job name
#SBATCH --mail-type=START,END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=frtim@robots.ox.ac.uk    # Where to send mail
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --mem=100gb                          # Job memory request
#SBATCH --time=128:59:00                     # Time limit hrs:min:sec
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:1                        # Requesting 2 GPUs
#SBATCH --output=/scratch/shared/beegfs/frtim/runs_v2/logs/out_org_obf_v4_8.txt
#SBATCH --error=/scratch/shared/beegfs/frtim/runs_v2/logs/error_org_obf_v4_8.txt
#SBATCH --constraint=m40             # Request nodes with specific features
# -------------------------------

source /users/frtim/.bashrc

module load cudnn/v8_11.3

conda activate env_subm20_is3

export GRADLE_USER_HOME=$tri_scratch_dir/runs_v2/gradles/gradles_org_obf_v4_8

mkdir $GRADLE_USER_HOME 

cd /users/frtim/Code/ma_minerl_training_original


/usr/bin/xvfb-run -s "-ac -screen 0 1280x1024x24" python train.py --env MineRLObtainDiamondDenseVectorObf-v0 --gpu 0 --outdir $tri_scratch_dir/runs_v2/out/org_obf_v4_8

