#!/bin/bash

#SBATCH --output=/home/%u/honours-project/contrastive-map/src/py/slurm_logs/slurm-%A_%a.out
#SBATCH --error=/home/%u/honours-project/contrastive-map/src/py/slurm_logs/slurm-err-%A_%a.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --cpus-per-task=4
#SBATCH --time=0-08:00:00
#SBATCH --partition=Teach-Short

START=$(date "+%d/%m/%Y %H:%M:%S")
echo "Job starting at ${START} on ${SLURM_JOB_NODELIST}"
STUDENT_ID=$(whoami)


# make available all commands
source /home/${STUDENT_ID}/.bashrc

echo "________________________________________"

echo "Using W&B API key: ${WANDB_API_KEY}"
echo "Running W&B in ${WANDB_MODE} mode"

# make script bail out after first error
# set -e

echo "________________________________________"

# define main directories
HOME_DIR=/home/${STUDENT_ID}/honours-project
SCRATCH_DIR=/disk/scratch_big/${STUDENT_ID}
EXPERIMENT_DIR=${HOME_DIR}/contrastive-map/src/py
DATA_DIR=${HOME_DIR}/contrastive-map/src/data/originals

# activate the virtual environment
echo "Activating virtual environment at ${HOME_DIR}/henv/bin/activate"
source ${HOME_DIR}/henv/bin/activate

echo "________________________________________"

# create scratch disk directory & any other directories which might not exist
echo "Creating directory in scratch disk: ${SCRATCH_DIR}"
mkdir -p ${SCRATCH_DIR}

SCRATCH_DATA_DIR=${SCRATCH_DIR}/data
SCRATCH_OUT_DIR=${SCRATCH_DIR}/output

EXPERIMENT_OUT_DIR=${EXPERIMENT_DIR}/output
SLURM_OUT_DIR=${EXPERIMENT_DIR}/slurm_logs

mkdir -p ${SCRATCH_DATA_DIR}
mkdir -p ${SCRATCH_OUT_DIR}
mkdir -p ${EXPERIMENT_OUT_DIR}
mkdir -p ${SLURM_OUT_DIR}

# see if anything is in out directory
ls ${SCRATCH_OUT_DIR}

echo "________________________________________"

# transfer the data file to scratch
echo "Transferring files from ${DATA_DIR} to ${SCRATCH_DATA_DIR}"
rsync --archive --update --compress --progress ${DATA_DIR}/ ${SCRATCH_DATA_DIR}

echo "________________________________________"

# run code
echo "Running main.py in ${EXPERIMENT_DIR}"
python ${EXPERIMENT_DIR}/main.py --batch-size 16 \
								 --patch-size 64 \
								 --epochs 1 \
								 --seed 23 \
								 --log-interval 1000 \
								 --input "${SCRATCH_DATA_DIR}" \
								 --output "${SCRATCH_OUT_DIR}" \
								 --byol-ema-tau 0.99

echo "________________________________________"


# transfer the data file from scratch
echo "Transferring files from ${SCRATCH_OUT_DIR} to ${EXPERIMENT_OUT_DIR}"
rsync --archive --update --compress --progress ${SCRATCH_OUT_DIR}/ ${EXPERIMENT_OUT_DIR}

echo "________________________________________"

END=$(date "+%d/%m/%Y %H:%M:%S")
echo "Job completed succesfully at ${END}"
