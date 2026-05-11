#!/bin/bash
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --job-name=training
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --array=0-1

sleep $((SLURM_ARRAY_TASK_ID * 30))

module load julia

PARAMS=(
    "trained_ace_class_ln10As_basis_mnuw0wacdm_10000 training_ln10As_basis.jl"
    "trained_ace_class_sigma8_basis_mnuw0wacdm_10000 training_sigma8_basis.jl"
)
PARAM_SET=(${PARAMS[$SLURM_ARRAY_TASK_ID]})


home_dir="/global/homes/j/jgmorawe/desi-emulators-pipeline"
scratch_dir="/pscratch/sd/j/jgmorawe"
path_input="${scratch_dir}/ace_class_mnuw0wacdm_10000"
path_output="${home_dir}/${PARAM_SET[0]}"
script_path="${home_dir}/ACE/codes/${PARAM_SET[1]}"
nn_setup_path="${home_dir}/ACE/supporting_files/nn_setup.json"
n_epoch=2000
n_run=20
batchsize=256


julia -t $SLURM_CPUS_PER_TASK  "$script_path" --path_input="$path_input" --path_output="$path_output" --nn_setup_path="$nn_setup_path" --n_epoch=$n_epoch --n_run=$n_run --batchsize=$batchsize