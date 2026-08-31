#!/bin/bash
#SBATCH --job-name=evalcov
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --exclude=node031,node032
#SBATCH --output=logs/eval_%j.out
#SBATCH --requeue
#
# The two evaluation steps, which go together and take a few minutes on
# GPU: eval_cov.py (NLL, calibration, figures) then denoise.py (residual
# projection, final MSE).
#
# LIVE IN THE TERMINAL:
#   srun --partition=short --time=00:30:00 --cpus-per-task=4 --mem=16G --gres=gpu:1 --exclude=node031,node032 bash evaluation.bash
#
# DEFERRED:
#   sbatch evaluation.bash
#
# ONLY TO BE LAUNCHED AFTER BOTH TRAININGS: the structured one AND the
# diagonal baseline. Without checkpoints/covdiag_best.pt, eval_cov.py still
# runs but warns, in large letters, that its numbers prove nothing.
#
# node031 and node032 are excluded for the same reason as in the other
# scripts: sm_61, incompatible with the torch of the dncnn env.

echo "Job started on:"
hostname
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dncnn

cd ~/structured_uncertainty_celeba

export CELEBA_DIR=$HOME/data/celeba/img_align_celeba
export CELEBA_CACHE=$HOME/data/celeba/celeba_64_gray.npy

# --------------------------------------------------------------------------
# GPU GUARD, the same one as in train_cov.bash.
#
# Added on 27 August after this job died in eight seconds on node023.
# On this cluster, a GPU already taken (exclusive mode) still lets
# torch.cuda.is_available() answer True and even displays the name of the
# card: those calls do not create a CUDA context. It is the FIRST allocation
# that fails -- here in the torch.load of the DnCNN -- with
#     CUDA error: CUDA-capable device(s) is/are busy or unavailable
#
# We put the job back in the queue rather than letting it die. It KEEPS ITS
# JOBID and will start again on another node. SLURM_RESTART_COUNT avoids the
# infinite loop if the problem is not an isolated node but an account limit.
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

# -u : unbuffered output, same reason as in train_dncnn.bash.
# main.py --evaluation chains the two scripts and stops at the first failure.
python -u main.py --evaluation "$@"
code=$?

# Python's exit code has to make it back up to SLURM. Without the "exit $code"
# below, the last "date" succeeds and the job is recorded as COMPLETED even
# if the training crashed: sacct then shows successes everywhere and you go
# hunting for the failure somewhere else. That is exactly what happened on
# 27 August with the six diagonal jobs that died in eight seconds.
echo "Job finished:"
date
exit $code
