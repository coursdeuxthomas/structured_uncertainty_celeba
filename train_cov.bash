#!/bin/bash
#SBATCH --job-name=cov
#SBATCH --partition=short
#SBATCH --time=01:55:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --exclude=node031,node032
#SBATCH --output=logs/cov_%j.out
#SBATCH --requeue
#
# Training of the covariance network, with the DnCNN frozen.
#
#   sbatch train_cov.bash                 structured model
#   sbatch train_cov.bash --diagonale     diagonal baseline
#
# BOTH RUNS ARE REQUIRED. The NLL gap between them is what puts a number on
# what the structure brings; without it the results prove nothing. They
# write under distinct names (cov_* and covdiag_*), so they can run one
# after the other or in parallel without stepping on each other.
#
# --resume is already on the command line below: relaunching this same
# script after the 1 h 55 cutoff carries on training where it left off.
# Count on several relaunches to get all the way through the 50 epochs.
#
# node031 and node032 are excluded for the same reason as in
# train_dncnn.bash: sm_61, incompatible with the torch of the dncnn env.

echo "Job started on:"
hostname
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dncnn

cd ~/structured_uncertainty_celeba

export CELEBA_DIR=$HOME/data/celeba/img_align_celeba
export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy

# --------------------------------------------------------------------------
# GPU GUARD, added on 27 August after a failure that took three seconds.
#
# On this cluster, a GPU already taken by another process (exclusive compute
# mode) still lets torch.cuda.is_available() answer True and even displays
# the name of the card: those calls do not create a CUDA context. It is the
# FIRST allocation that fails, with
#     CUDA error: CUDA-capable device(s) is/are busy or unavailable
#
# Without this test, the job dies in three seconds and the failure spreads
# instantly to the whole --dependency=afterany chain: six jobs gone in twenty
# seconds. Instead, we put the job back in the queue. It KEEPS ITS JOBID, so
# the chain behind it stays intact, and it will start again on another
# node.
#
# The SLURM_RESTART_COUNT counter avoids the infinite loop if the problem is
# not an isolated node but an account limit.
# --------------------------------------------------------------------------
if ! python -c "import torch; torch.zeros(1, device='cuda')" 2>/dev/null; then
  echo "GPU inutilisable sur $(hostname) : deja occupe, ou mode exclusif."
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 5 ]; then
    echo "Remise en file du job $SLURM_JOB_ID (tentative ${SLURM_RESTART_COUNT:-0})."
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60
    exit 0
  fi
  echo "Cinq tentatives ont echoue : ce n'est pas un noeud isole. On s'arrete."
  exit 1
fi
echo "GPU utilisable, demarrage."

# -u : unbuffered output, otherwise the log stays empty for several minutes.
python -u train_cov.py --epochs 50 --resume "$@"
code=$?

# Python's exit code has to make it back up to SLURM. Without the "exit $code"
# below, the last "date" succeeds and the job is recorded as COMPLETED even
# if the training crashed: sacct then shows successes everywhere and you go
# hunting for the failure somewhere else. That is exactly what happened on
# 27 August with the six diagonal jobs that died in eight seconds.
echo "Job finished:"
date
exit $code
