#!/bin/bash

#SBATCH -N {N}
#SBATCH -c {c}
#SBATCH -n {n}
#SBATCH --mem={m}G
#SBATCH -p {p}
#SBATCH -q {q}
#SBATCH -t 0-{t}:00:00
#SBATCH -o ./slurms/%j.{o}
#SBATCH --job-name={j}

# Load required modules
module load gcc-11.2.0-gcc-11.2.0
module load openmpi-4.1.3-gcc-11.2.0

# Run for the task
{cgd}
srun {cmd}

