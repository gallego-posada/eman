#!/bin/bash

# This script provide simple commands illustrating how to trigger FAUST and TOSCA experiments
# To be ran from ./eman/

declare -a model_types=("EMAN" "RelTanEMAN" "GemCNN" "RelTanGemCNN")

for model in "${model_types[@]}"; do

    echo "Running ${model} on FAUST"
    python experiments/faust_direct.py -equiv_bias --model ${model} --epochs 1 --seed 0
    
    echo "Running ${model} on TOSCA"
    python experiments/tosca_direct.py --model ${model} --epochs 1 --seed 5

done
