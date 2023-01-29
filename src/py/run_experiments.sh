# take in user input
N_EXPERIMENTS=$1
MAX_PARALLEL_JOBS=$2

# gather the files & directories to use
STUDENT_ID=$(whoami)
EXPERIMENT_DIR=/home/${STUDENT_ID}/honours-project/contrastive-map/src/py
MAIN_FILE=${EXPERIMENT_DIR}/main.py 
SLURM_RUN_FILE=${EXPERIMENT_DIR}/batch_slurm_contrastive_run.sh
GEN_EXPERIMENT_FILE=${EXPERIMENT_DIR}/generate_experiments.py
EXPERIMENT_FILE=${EXPERIMENT_DIR}/experiments.txt

# generate experiments file
echo "Generating experiments.txt"
python ${GEN_EXPERIMENT_FILE} -i ${MAIN_FILE}

# run sbatch job
echo "Running batch job: sbatch --array=1-${N_EXPERIMENTS}%${MAX_PARALLEL_JOBS} ${SLURM_RUN_FILE} ${EXPERIMENT_FILE}"
sbatch --array=1-${N_EXPERIMENTS}%${MAX_PARALLEL_JOBS} ${SLURM_RUN_FILE} ${EXPERIMENT_FILE}