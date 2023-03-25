#!/bin/bash

#SBATCH --job-name=canonical
#SBATCH --output=/home/%u/honours-project/contrastive-map/src/py/slurm_logs/canonical_logs/slurm-%A_%a.out
#SBATCH --error=/home/%u/honours-project/contrastive-map/src/py/slurm_logs/canonical_logs/slurm-err-%A_%a.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=2
#SBATCH --time=3-08:00:00
#SBATCH --partition=PGR-Standard
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=s1908368@ed.ac.uk

START=$(date "+%d/%m/%Y %H:%M:%S")
echo "Job starting at ${START} on ${SLURMD_NODENAME}"
STUDENT_ID=$(whoami)


# make available all commands
source /home/${STUDENT_ID}/.bashrc

# make script bail out after first error
set -e

echo "________________________________________"

# define main directories
if [ -z "${HOME_DIR}" ]; then
  HOME_DIR=/home/${STUDENT_ID}/honours-project
fi

if [ -z "${SCRATCH_DIR}" ]; then
  SCRATCH_DIR=/disk/scratch_big/${STUDENT_ID}
fi

if [ -z "${EXPERIMENT_DIR}" ]; then
  EXPERIMENT_DIR=${HOME_DIR}/contrastive-map/src/py
fi

if [ -z "${EXPERIMENT_FILE}" ]; then
  EXPERIMENT_FILE=${EXPERIMENT_DIR}/experiments.txt
fi

if [ -z "${EXPERIMENT_NAME}" ]; then
  EXPERIMENT_NAME=${SLURM_JOB_ID}
fi

DATA_DIR=${HOME_DIR}/contrastive-map/src/data/originals

if [ -f "${EXPERIMENT_FILE}" ]; then
    echo "${EXPERIMENT_FILE} found."
else
	echo "${EXPERIMENT_FILE} wasn't found. Run generate_canonical_experiments.txt!"
	exit
fi

echo "________________________________________"

# activate the virtual environment
echo "Activating virtual environment at ${HOME_DIR}/henv/bin/activate"
source ${HOME_DIR}/henv/bin/activate

echo "________________________________________"

# create scratch disk directory & any other directories which might not exist
echo "Creating directory in scratch disk: ${SCRATCH_DIR}"
mkdir -p ${SCRATCH_DIR}

if [ -z "${SCRATCH_DATA_DIR}" ]; then
  SCRATCH_DATA_DIR=${SCRATCH_DIR}/data
fi

if [ -z "${SCRATCH_CANONICAL_DATA_DIR}" ]; then
  SCRATCH_CANONICAL_DATA_DIR=${SCRATCH_DIR}/data/osm_carto
fi

if [ -z "${SCRATCH_OUT_DIR}" ]; then
  SCRATCH_OUT_DIR=${SCRATCH_DIR}/output/${EXPERIMENT_NAME}
fi

rm -rf ${SCRATCH_DATA_DIR}
rm -rf ${SCRATCH_CANONICAL_DATA_DIR}
rm -rf ${SCRATCH_OUT_DIR}

BEST_BYOL_DIR=b-presnet18-e25-b64-t0_80-lr0_001-p128
BEST_SIMCLR_DIR=s-cnn-e25-b64-t0_99-lr0_001-p128

SCRATCH_CHECKPOINT_DIR_SIMCLR=${SCRATCH_OUT_DIR}/${BEST_SIMCLR_DIR}
SCRATCH_CHECKPOINT_DIR_BYOL=${SCRATCH_OUT_DIR}/${BEST_BYOL_DIR}

EXPERIMENT_OUT_DIR=${EXPERIMENT_DIR}/output/${EXPERIMENT_NAME}

EXPERIMENT_CHECKPOINT_DIR_SIMCLR=${EXPERIMENT_OUT_DIR}/${BEST_SIMCLR_DIR}
EXPERIMENT_CHECKPOINT_DIR_BYOL=${EXPERIMENT_OUT_DIR}/${BEST_BYOL_DIR}

SLURM_OUT_DIR=${EXPERIMENT_DIR}/slurm_logs/canonical_logs

mkdir -p ${SCRATCH_DATA_DIR}
mkdir -p ${SCRATCH_CANONICAL_DATA_DIR}
mkdir -p ${SCRATCH_OUT_DIR}
mkdir -p ${EXPERIMENT_OUT_DIR}
mkdir -p ${SLURM_OUT_DIR}

mkdir -p ${SCRATCH_CHECKPOINT_DIR_SIMCLR}
mkdir -p ${SCRATCH_CHECKPOINT_DIR_BYOL}

#rm -rf ${SCRATCH_DATA_DIR}/*.pk ||:

if [[ "$DEBUG" == "true" ]]; then
  echo "Debugging set to true"
  echo "Home directory: ${HOME_DIR}"
  echo "Scratch directory: ${SCRATCH_DIR}"
  echo "Scratch data directory: ${SCRATCH_DATA_DIR}"
  echo "Scratch canonical data directory: ${SCRATCH_CANONICAL_DATA_DIR}"
  echo "Scratch output directory: ${SCRATCH_OUT_DIR}"
  echo "Scratch SimCLR checkpoint": ${SCRATCH_CHECKPOINT_DIR_SIMCLR}
  echo "Scratch BYOL checkpoint": ${SCRATCH_CHECKPOINT_DIR_BYOL}
  echo "Experiment directory: ${EXPERIMENT_DIR}"
  echo "Experiment output directory: ${EXPERIMENT_OUT_DIR}"
  echo "Experiment SimCLR checkpoint": ${EXPERIMENT_CHECKPOINT_DIR_SIMCLR}
  echo "Experiment BYOL checkpoint": ${EXPERIMENT_CHECKPOINT_DIR_BYOL}
  echo "Data directory: ${DATA_DIR}"
  echo "Experiment file: ${EXPERIMENT_FILE}"
  echo "Experiment name: ${EXPERIMENT_NAME}"

  exit
fi

echo "________________________________________"

# transfer the data file to scratch
echo "Transferring files from ${DATA_DIR} to ${SCRATCH_DATA_DIR}"
rsync --archive --update --compress --progress ${DATA_DIR}/ ${SCRATCH_DATA_DIR}

echo "Transferring SimCLR checkpoint from ${EXPERIMENT_CHECKPOINT_DIR_SIMCLR} to ${SCRATCH_CHECKPOINT_DIR_SIMCLR}"
rsync --archive --update --compress --progress ${EXPERIMENT_CHECKPOINT_DIR_SIMCLR}/ ${SCRATCH_CHECKPOINT_DIR_SIMCLR}

echo "Transferring BYOL checkpoint from ${EXPERIMENT_CHECKPOINT_DIR_BYOL} to ${SCRATCH_CHECKPOINT_DIR_BYOL}"
rsync --archive --update --compress --progress ${EXPERIMENT_CHECKPOINT_DIR_BYOL}/ ${SCRATCH_CHECKPOINT_DIR_BYOL}

echo "________________________________________"

# run code
echo "Running canonical_main.py according to experiments from ${EXPERIMENT_FILE}"
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${EXPERIMENT_FILE}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully."

echo "________________________________________"


# transfer the output file from scratch
echo "Transferring files from ${SCRATCH_OUT_DIR} to ${EXPERIMENT_OUT_DIR}"
rsync --archive --update --compress --progress ${SCRATCH_OUT_DIR}/ ${EXPERIMENT_OUT_DIR}/canonical

echo "________________________________________"

END=$(date "+%d/%m/%Y %H:%M:%S")
echo "Job completed successfully at ${END}"
