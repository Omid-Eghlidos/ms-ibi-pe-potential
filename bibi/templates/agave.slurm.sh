#!/bin/bash

#SBATCH -N {N}
#SBATCH -c {c}
#SBATCH -n {n}
#SBATCH --mem={m}G
#SBATCH -p {p}
#SBATCH -q {q}
#SBATCH -t 0-{t}:00:00
#SBATCH -o {o}
#SBATCH --job-name={j}

# Load required modules
module purge
module load gcc/8.2.0

# Run for the task
{cgd}
mpiexec {cmd}

