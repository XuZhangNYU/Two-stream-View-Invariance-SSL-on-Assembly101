#!/bin/bash
#SBATCH --job-name=fsl_3dsoft_echo_3_epoch_full_Data
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/xz4863/scratch/sync_test_run.out
#SBATCH --error=/home/xz4863//scratch/sync_test_run.err
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --account=pr_113_tandon_advanced
#SBATCH --cpus-per-task=8  # Request 8 CPU cores
#SBATCH --ntasks=1  # Ensures a single task (common for data parallel jobs)
#SBATCH --nodes=1   # Keeps everything on one node
#SBATCH --mem=128G   # Request 32GB RAM
#SBATCH --time=6:00:08
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xz4863@nyu.edu

export SSL_CERT_FILE=/scratch/xz4863/certificate_for_WANDB/cacert.pem

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro   /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif   /bin/bash -c "source /ext3/miniconda3/bin/activate synchranization && python3 /home/xz4863/scratch/o_c2f/main_fsl_with_3dsoftmax.py \
    --action train \
    --feature_path /scratch/xz4863/TSM_features \
    --split train \
    > /scratch/xz4863/synchranization_all_u_need_results/run_log/v3_train_run_log.txt"