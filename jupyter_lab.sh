#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=Teach-Short
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --mem=16GB
#SBATCH --output=/home/%u/honours-project/contrastive-map/logs/jupyter.log

STUDENT_ID=$(whoami)
HOME_DIR=/home/${STUDENT_ID}/honours-project
EXPERIMENT_DIR=${HOME_DIR}/contrastive-map/src/dev

source ${HOME_DIR}/henv/bin/activate

cd ${EXPERIMENT_DIR}

JUPYTER_PORT=8888

echo "Running jupyter on $(pwd), at port ${JUPYTER_PORT}"

jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT}
