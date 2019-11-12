#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=perceptual-untangling-bootstrap
#SBATCH --output=slurm-output/%j.out
#SBATCH --mail-user=ojh221@nyu.edu

cd $HOME/perceptual-straightening
sh analyzeGlobal/submit.sh

# leave a blank line at the end