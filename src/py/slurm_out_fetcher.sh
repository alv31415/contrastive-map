#!/bin/bash

#SBATCH --output=/home/%u/honours-project/contrastive-map/src/py/slurm_logs/slurm-fetch-%A_%a.out
#SBATCH --error=/home/%u/honours-project/contrastive-map/src/py/slurm_logs/slurm-fetch-err-%A_%a.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:30:00
#SBATCH --partition=Teach-Short

START=$(date "+%d/%m/%Y %H:%M:%S")
echo "Fetching job starting at ${START} on ${SLURM_JOB_NODELIST}"
STUDENT_ID=$(whoami)


# make available all commands
source /home/${STUDENT_ID}/.bashrc

echo "________________________________________"

# define main directories
HOME_DIR=/home/${STUDENT_ID}/honours-project
SCRATCH_DIR=/disk/scratch_big/${STUDENT_ID}
EXPERIMENT_DIR=${HOME_DIR}/contrastive-map/src/py
DATA_DIR=${HOME_DIR}/contrastive-map/src/data/originals

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
ls ${SCRATCH_DATA_DIR}

echo "________________________________________"


# transfer the output file from scratch
echo "Transferring files from ${SCRATCH_OUT_DIR} to ${EXPERIMENT_OUT_DIR}"
rsync --archive --update --compress --progress ${SCRATCH_OUT_DIR}/ ${EXPERIMENT_OUT_DIR}
rsync --archive --update --compress --progress ${SCRATCH_DATA_DIR}/patch_dataset.pk ${EXPERIMENT_OUT_DIR}

echo "________________________________________"

END=$(date "+%d/%m/%Y %H:%M:%S")
echo "Job completed succesfully at ${END}"
