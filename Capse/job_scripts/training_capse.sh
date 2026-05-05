#!/bin/bash
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --job-name=training
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-3

sleep $((SLURM_ARRAY_TASK_ID * 30))

module load julia

PARAMS=(
    "TT"
    "EE"
    "TE"
    "PP"
)

spectrum="${PARAMS[$SLURM_ARRAY_TASK_ID]}"

home_dir="/global/homes/j/jgmorawe/desi-emulators-pipeline"
scratch_dir="/pscratch/sd/j/jgmorawe"
path_input="${scratch_dir}/capse_class_mnuw0wacdm_10000"
path_output="${home_dir}/trained_capse_class_mnuw0wacdm_10000"
nn_setup_path="${home_dir}/Capse/supporting_files/nn_setup.json"
n_epoch=2000
n_run=20
batchsize=512

julia -t $SLURM_CPUS_PER_TASK  "${home_dir}/Capse/codes/training.jl" --spectrum="$spectrum" --path_input="$path_input" --path_output="$path_output" --nn_setup_path="$nn_setup_path" --n_epoch=$n_epoch --n_run=$n_run --batchsize=$batchsize