# make script bail out after first error
set -e

# gather the files & directories to use
DEBUG=true
STUDENT_ID=$(whoami)
HOME_DIR=/home/${STUDENT_ID}/honours-project
EXPERIMENT_DIR=${HOME_DIR}/contrastive-map/src/py
EXPERIMENT_NAME=final_run

SCRATCH_DIR=/disk/scratch_big/${STUDENT_ID}
SCRATCH_DATA_DIR=${SCRATCH_DIR}/data
SCRATCH_OUT_DIR=${SCRATCH_DIR}/output/${EXPERIMENT_NAME}

MAIN_FILE=${EXPERIMENT_DIR}/main.py 
SLURM_RUN_FILE=${EXPERIMENT_DIR}/batch_slurm_contrastive_run.sh
GEN_EXPERIMENT_FILE=${EXPERIMENT_DIR}/generate_experiments.py
EXPERIMENT_FILE=${EXPERIMENT_DIR}/experiments.txt

# generate experiments file
echo "Generating experiments.txt"
python ${GEN_EXPERIMENT_FILE} --main ${MAIN_FILE} \
                              --scratch-out-dir ${SCRATCH_OUT_DIR} \
							                --scratch-data-dir ${SCRATCH_DATA_DIR}

echo "experiments.txt created successfully"

N_EXPERIMENTS=$(cat ${EXPERIMENT_FILE} | wc -l) 
MAX_PARALLEL_JOBS=8

echo "${N_EXPERIMENTS} found. Running with maximum ${MAX_PARALLEL_JOBS} parallel jobs."

# run sbatch job
echo "Running batch job: sbatch --array=1-${N_EXPERIMENTS}%${MAX_PARALLEL_JOBS} ${SLURM_RUN_FILE} ${EXPERIMENT_FILE}"
sbatch --array=1-${N_EXPERIMENTS}%${MAX_PARALLEL_JOBS}  \
       --export=HOME_DIR="${HOME_DIR}" \
       --export=EXPERIMENT_DIR="${EXPERIMENT_DIR}" \
       --export=SCRATCH_DIR="${SCRATCH_DIR}" \
       --export=SCRATCH_DATA_DIR="${SCRATCH_DATA_DIR}" \
       --export=SCRATCH_OUT_DIR="${SCRATCH_OUT_DIR}" \
       --export=EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
       --export=EXPERIMENT_FILE="${EXPERIMENT_FILE}" \
       --export=DEBUG=${DEBUG} \
       ${SLURM_RUN_FILE}
