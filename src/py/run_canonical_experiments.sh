# make script bail out after first error
set -e

# gather the files & directories to use
export DEBUG=false
STUDENT_ID=$(whoami)
export HOME_DIR=/home/${STUDENT_ID}/honours-project
export EXPERIMENT_DIR=${HOME_DIR}/contrastive-map/src/py
export EXPERIMENT_NAME=final_run_2

export SCRATCH_DIR=/disk/scratch_big/${STUDENT_ID}
export SCRATCH_DATA_DIR=${SCRATCH_DIR}/data
export SCRATCH_CANONICAL_DATA_DIR=${SCRATCH_DATA_DIR}/osm_carto
export SCRATCH_OUT_DIR=${SCRATCH_DIR}/output/${EXPERIMENT_NAME}

MAIN_FILE=${EXPERIMENT_DIR}/canonical_main.py
SLURM_RUN_FILE=${EXPERIMENT_DIR}/batch_slurm_canonical_run.sh
GEN_EXPERIMENT_FILE=${EXPERIMENT_DIR}/generate_canonical_experiments.py
export EXPERIMENT_FILE=${EXPERIMENT_DIR}/canonical_experiments.txt

# generate experiments file
echo "Generating canonical_experiments.txt"
python ${GEN_EXPERIMENT_FILE} --main ${MAIN_FILE} \
							  --scratch-out-dir ${SCRATCH_OUT_DIR} \
							  --scratch-data-dir ${SCRATCH_CANONICAL_DATA_DIR} \
							  --patch-dataset-dir ${SCRATCH_DATA_DIR}

echo "canonical_experiments.txt created succesfully"

N_EXPERIMENTS=$(cat ${EXPERIMENT_FILE} | wc -l)
MAX_PARALLEL_JOBS=14

echo "________________________________________"

echo "Home directory: ${HOME_DIR}"
echo "Experiment directory: ${EXPERIMENT_DIR}"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Scratch directory: ${SCRATCH_DIR}"
echo "Scratch data directory: ${SCRATCH_DATA_DIR}"
echo "Scratch output directory: ${SCRATCH_OUT_DIR}"
echo "Scratch canonical data directory: ${SCRATCH_CANONICAL_DATA_DIR}"

echo "________________________________________"

echo "${N_EXPERIMENTS} found. Running with maximum ${MAX_PARALLEL_JOBS} parallel jobs."

# run sbatch job
echo "Running batch job: sbatch --array=1-${N_EXPERIMENTS}%${MAX_PARALLEL_JOBS} ${SLURM_RUN_FILE} ${EXPERIMENT_FILE}"
sbatch --array=1-${N_EXPERIMENTS}%${MAX_PARALLEL_JOBS} --export=ALL ${SLURM_RUN_FILE}
