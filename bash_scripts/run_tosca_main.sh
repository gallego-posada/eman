#!/usr/bin/bash

source ~/.bashrc
module load anaconda
conda activate eman

python ~/eman/experiments/tosca_direct.py "$@"