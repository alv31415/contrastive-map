# make script bail out after first error
set -e

# gather the files & directories to use
STUDENT_ID=$(whoami)
EXPERIMENT_DIR=/home/${STUDENT_ID}/honours-project/contrastive-map/src/py/canonical_representation

SCRATCH_DIR=/disk/scratch_big/${STUDENT_ID}
SCRATCH_PATCH_DATA_DIR=${SCRATCH_DIR}/data
SCRATCH_CANONICAL_DATA_DIR=${SCRATCH_DIR}/data/osm_carto
SCRATCH_OUT_DIR=${SCRATCH_DIR}/output

MAIN_FILE=${EXPERIMENT_DIR}/canonical_main.py
SLURM_RUN_FILE=${EXPERIMENT_DIR}/batch_slurm_canonical_run.sh
GEN_EXPERIMENT_FILE=${EXPERIMENT_DIR}/generate_canonical_experiments.py
EXPERIMENT_FILE=${EXPERIMENT_DIR}/canonical_experiments.txt

# generate experiments file
echo "Generating canonical_experiments.txt"
python ${GEN_EXPERIMENT_FILE} --main ${MAIN_FILE} \
							  --scratch-out-dir ${SCRATCH_OUT_DIR} \
							  --scratch-data-dir ${SCRATCH_CANONICAL_DATA_DIR} \
							  --patch-dataset-dir ${SCRATCH_PATCH_DATA_DIR}

echo "canonical_experiments.txt created succesfully"


N_EXPERIMENTS=$(cat ${EXPERIMENT_FILE} | wc -l)
MAX_PARALLEL_JOBS=8

echo "${N_EXPERIMENTS} found. Running with maximum ${MAX_PARALLEL_JOBS} parallel jobs."

# run sbatch job
echo "Running batch job: sbatch --array=1-${N_EXPERIMENTS}%${MAX_PARALLEL_JOBS} ${SLURM_RUN_FILE} ${EXPERIMENT_FILE}"
sbatch --array=1-${N_EXPERIMENTS}%${MAX_PARALLEL_JOBS} ${SLURM_RUN_FILE}