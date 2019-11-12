#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --mem=1GB
#SBATCH --job-name=perceptual-untangling-bootstrap
#SBATCH --output=slurm-output/%j.out
#SBATCH --mail-user=ojh221@nyu.edu


module purge
module load torch/gnu/20170504
# module load torch/intel/20170104
cd $HOME/perceptual-straightening
th analyzeGlobal/bootstrap.lua -seed -1                   -domain perceptual -dim 10 -repeatNumber $1

# leave a blank line at the end