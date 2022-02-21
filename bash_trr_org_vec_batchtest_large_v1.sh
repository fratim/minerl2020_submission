#!/bin/bash
#SBATCH --job-name=org_tree_vec_batchtest_large_v1_3                   # Job name
#SBATCH --mail-type=START,END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=frtim@robots.ox.ac.uk    # Where to send mail
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --mem=60gb                          # Job memory request
#SBATCH --time=167:59:00                     # Time limit hrs:min:sec
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:1                        # Requesting 2 GPUs
#SBATCH --output=/scratch/shared/beegfs/frtim/runs_v2/logs/out_org_tree_vec_batchtest_large_v1_3.txt
#SBATCH --error=/scratch/shared/beegfs/frtim/runs_v2/logs/error_org_tree_vec_batchtest_large_v1_3.txt
#SBATCH --constraint=p40             # Request nodes with specific features
# -------------------------------

source /users/frtim/.bashrc

module load cudnn/v8_11.3

conda activate env_subm20_is2

export GRADLE_USER_HOME=$tri_scratch_dir/runs_v2/gradles/gradles_org_tree_vec_batchtest_large_v1_3

mkdir $GRADLE_USER_HOME 

cd /users/frtim/Code_remote/ma_minerl_vec

/usr/bin/xvfb-run -s "-ac -screen 0 1280x1024x24" python train.py --env MineRLTreechopVectorObf-v0 --gpu 0 --outdir $tri_scratch_dir/runs_v2/out/org_tree_vec_batchtest_large_v1_3 --seed 11 --batch-size 3200 --update-interval 400
