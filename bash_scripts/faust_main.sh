#!/usr/bin/bash

# To be ran from eman folder

main_script="$HOME/eman/bash_scripts/run_faust_main.sh"
slurm_log_dir="$HOME/eman/experiment_outputs/faust_main_exp_logs"
notify_email=""
partition="gpu"

epochs="100"

# WandB configs
use_wandb=True
wandb_offline=True
wb_tag="faust_main_exp"

# Not applying roto_translations
other_flags="-random_test_gauge -equiv_bias -save_model_dict"

# Declare arrays for looping over
declare -a use_reltan=("" "RelTan")
declare -a model_types=("GemCNN" "EMAN")
declare -a seeds=(1 2 3 4 5)

# Set up wandb flags
wb_cmd=""
if [ "$use_wandb" = "True" ]; then wb_cmd="$wb_cmd -wandb --wb_tag $wb_tag"; fi
if [ "$wandb_offline" = "True" ]; then wb_cmd="$wb_cmd -wandb_offline"; fi

for seed in ${seeds[@]}; do
    for model in "${model_types[@]}"; do
        for reltan in "${use_reltan[@]}"; do

            model_str="$reltan$model"

            if [ "$reltan" = "RelTan" ]; then
                # Only RelTan models use rel_power_list
                declare -a rpls=(".5 .7" ".7")
                dp_cmd="--deg_power 1.5"; dp_job="-deg_power-1.5"
            else
                declare -a rpls=("")
                dp_cmd=""; dp_job=""
            fi

            for rpl in "${rpls[@]}"; do

                if [ "${rpl}" = "" ]; then
                    rpl_cmd=""; rpl_job=""
                else
                    rpl_cmd="--rel_power_list ${rpl}"
                    rpl_job="-rpl-${rpl// /}"
                fi

                job_name="${wb_tag}-model-${model_str}-epochs-${epochs}\
                          -seed-${seed}${rpl_job}${dp_job}"
                clean_job_name=${job_name// /}
                echo ${clean_job_name}

                cmd_line="${wb_cmd} ${other_flags} --model ${model_str}\
                          --epochs ${epochs} --seed ${seed} ${rpl_cmd} ${dp_cmd}"
                echo ${cmd_line}

                sbatch --job-name eman-$clean_job_name --time=2:00:00 \
                --cpus-per-task 8 --mem=16G --gres=gpu:1 --partition=$partition\
                --output $slurm_log_dir/$clean_job_name.out \
                --mail-type=ALL --mail-user=$notify_email \
                $main_script $cmd_line

            done
        done
    done
done