#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --nodelist=landonia24
#SBATCH --partition=Teach-LongJobs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=0-09:00:00
#SBATCH --mem=16GB
#SBATCH --output=/home/%u/honours-project/contrastive-map/src/jupyter_logs/jupyter.log

STUDENT_ID=$(whoami)
HOME_DIR=/home/${STUDENT_ID}/honours-project
EXPERIMENT_DIR=${HOME_DIR}/contrastive-map/src/dev
SCRATCH_DIR=/disk/scratch_big/${STUDENT_ID}
SCRATCH_OUT_DIR=${SCRATCH_DIR}/output

JUPYTER_DIR=${SCRATCH_DIR}

source ${HOME_DIR}/henv/bin/activate

cd ${JUPYTER_DIR}

JUPYTER_PORT=8000

echo "Running jupyter on $(pwd), at port ${JUPYTER_PORT}"

jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser
