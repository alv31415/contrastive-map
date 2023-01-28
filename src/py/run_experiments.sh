# take in user input
N_EXPERIMENTS=$1
MAX_PARALLEL_JOBS=$2
EXPERIMENT_FILE=experiments.txt

STUDENT_ID=$(whoami)
MAIN_DIR=/home/${STUDENT_ID}/honours-project/contrastive-map/src/py/main.py 

# generate experiments file
echo "Generating experiments.txt"
python generate_experiments.py -i ${MAIN_DIR}

# run sbatch job
echo "Running batch job: sbatch --array=1-${N_EXPERIMENTS}%${MAX_PARALLEL_JOBS} batch_slurm_contrastive_run.sh ${EXPERIMENT_FILE}"
sbatch --array=1-${N_EXPERIMENTS}%${MAX_PARALLEL_JOBS} batch_slurm_contrastive_run.sh ${EXPERIMENT_FILE}