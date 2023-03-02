#!/bin/bash

#SBATCH --output=/home/%u/honours-project/contrastive-map/src/py/slurm_logs/canonical_logs/slurm-%A_%a.out
#SBATCH --error=/home/%u/honours-project/contrastive-map/src/py/slurm_logs/canonical_logs/slurm-err-%A_%a.out
#SBATCH --nodes=1
#SBATCH --nodelist=landonia23
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --time=3-08:00:00
#SBATCH --partition=Teach-LongJobs
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=s1908368@ed.ac.uk

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
DATA_DIR=${HOME_DIR}/contrastive-map/src/data/originals/osm_carto
EXPERIMENT_FILE=${EXPERIMENT_DIR}/canonical_experiments.txt

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

SCRATCH_DATA_DIR=${SCRATCH_DIR}/data/osm_carto
SCRATCH_OUT_DIR=${SCRATCH_DIR}/output

SCRATCH_CHECKPOINT_DIR=${SCRATCH_OUT_DIR}/b-presnet18-e5-b32-t0_9-p64

EXPERIMENT_OUT_DIR=${EXPERIMENT_DIR}/output

EXPERIMENT_CHECKPOINT_DIR=${EXPERIMENT_OUT_DIR}/b-presnet18-e5-b32-t0_9-p64

SLURM_OUT_DIR=${EXPERIMENT_DIR}/slurm_logs/canonical_logs

mkdir -p ${SCRATCH_DATA_DIR}
mkdir -p ${SCRATCH_OUT_DIR}
mkdir -p ${EXPERIMENT_OUT_DIR}
mkdir -p ${SLURM_OUT_DIR}

#rm -rf ${SCRATCH_DATA_DIR}/*.pk ||:

echo "________________________________________"

# transfer the data file to scratch
echo "Transferring files from ${DATA_DIR} to ${SCRATCH_DATA_DIR}"
rsync --archive --update --compress --progress ${DATA_DIR}/ ${SCRATCH_DATA_DIR}

echo "Transferring checkpoint from ${DATA_DIR} to ${SCRATCH_DATA_DIR}"
rsync --archive --update --compress --progress ${EXPERIMENT_CHECKPOINT_DIR}/ ${SCRATCH_CHECKPOINT_DIR}

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
rsync --archive --update --compress --progress ${SCRATCH_OUT_DIR}/ ${EXPERIMENT_OUT_DIR}

echo "________________________________________"

END=$(date "+%d/%m/%Y %H:%M:%S")
echo "Job completed successfully at ${END}"
