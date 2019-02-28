#!/bin/sh
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=convolution
#SBATCH --mail-type=END
#SBATCH --mail-user=sk7090@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.6.3
module load zlib/intel/1.2.8
module load intel/17.0.1

cd $SCRATCH

srun python3 -u "pyramid2x2.py" > output2x2.out
