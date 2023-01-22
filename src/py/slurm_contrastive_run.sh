#!/bin/bash

#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH --error=/home/%u/slurm_logs/slurm-err-%A_%a.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:05:00
#SBATCH --partition=PGR-Standard

START=$(date "+%d/%m/%Y %H:%M:%S")
echo "Job starting at ${START} on ${SLURM_JOB_NODELIST}"

STUDENT_ID=$(whoami)

# make available all commands
source /home/${STUDENT_ID}/.bashrc

# make script bail out after first error
set -e

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

# create scratch disk directory
echo "Creating directory in scratch disk: ${SCRATCH_DIR}"
mkdir -p ${SCRATCH_DIR}

SCRATCH_DATA_DIR=${SCRATCH_DIR}/data
SCRATCH_OUT_DIR=${SCRATCH_DIR}/output

echo "________________________________________"

# transfer the data file to scratch
echo "Transferring files from ${DATA_DIR} to ${SCRATCH_DATA_DIR}"
rsync --archive --verbose --update --compress --progress ${DATA_DIR}/ ${SCRATCH_DATA_DIR}

echo "________________________________________"

# run code
echo "Running main.py in ${EXPERIMENT_DIR}"
python ${EXPERIMENT_DIR}/main.py --batch-size 50 \
								 --patch-size 64 \
								 --epochs 5 \
								 --seed 23 \
								 --log-interval 20 \
								 --input "${SCRATCH_DATA_DIR}" \
								 --output "${SCRATCH_OUT_DIR}" \
								 --byol-ema-tau 0.99

echo "________________________________________"

EXPERIMENT_OUT_DIR=${EXPERIMENT_DIR}/output

# transfer the data file from scratch
echo "Transferring files from ${SCRATCH_OUT_DIR} to ${EXPERIMENT_OUT_DIR}"
rsync --archive --verbose --update --compress --progress ${SCRATCH_OUT_DIR}/ ${EXPERIMENT_OUT_DIR}

echo "________________________________________"

END=$(date "+%d/%m/%Y %H:%M:%S")
echo "Job completed succesfully at ${END}"
