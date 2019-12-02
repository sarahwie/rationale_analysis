bash Rationale_Analysis/commands/model_a_train_script.sh

sbatch --time=$( t2sd ) Cluster_scripts/multi_gpu_sbatch.sh bash Rationale_Analysis/commands/direct_script.sh

bash Rationale_Analysis/commands/direct_script_crf.sh