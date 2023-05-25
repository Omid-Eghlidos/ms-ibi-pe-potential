#!/bin/bash

LMP_EXE=/home/oeghlido/lmp_g++_openmpi

# Runs a job script.
# Input arguments:
#   $1 system number
#   $2 wall time
#   $3 job name
#   $4 lammps input file
#   $5 temperature

function slurm () {
  echo \
"#!/bin/bash
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH -p general
#SBATCH -q public
#SBATCH -t 0-$2:00:00
#SBATCH -o $1.$3.slurm
#SBATCH --job-name=$3-$1
module load gcc-8.5.0-gcc-11.2.0
module load openmpi-4.1.3-gcc-11.2.0
srun $LMP_EXE -in $4 -l logs/$1.$3.log -v sys $1 -v T $5" | sbatch
}

# Create output directories if they don't already exist.
mkdir -p logs samples slurms

# Move the slurm files into the ./slurms folder
for SLURM in *.slurm; do
    if [ -f ${SLURM} ]; then
        mv ${SLURM} ./slurms
    fi
done

# Loop over all data files and run the first case.
for f in ../data/*.data; do
    sys=`echo $f | cut -d "/" -f 3 | cut -d "." -f 1`
    w1=`echo $(pwd) | cut -d "/" -f 8 | cut -d "_" -f 1`
    w2=`echo $(pwd) | cut -d "/" -f 8 | cut -d "_" -f 2`
    echo "For system ${sys}."
    for T in `seq 300 20 500`  ; do
        if [ ! -f samples/${sys}.${T} ]; then
            echo "Submit the sampling job for ${T} K."
            slurm ${sys} 4 ${T}$w1$w2 ../diffusion.in ${T}
        fi
    printf "\n"
    done
done

