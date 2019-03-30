#!/usr/bin/env bash
#!/bin/sh
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=c_all_49
#SBATCH --mail-type=END
#SBATCH --mail-user=sk7090@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load seaborn/0.7.1
module swap python/intel python3/intel/3.5.3
module load jupyter-kernels/py3.5

cd $SCRATCH

srun python3 -u "neural network all digits.py" > output_all_49.out
