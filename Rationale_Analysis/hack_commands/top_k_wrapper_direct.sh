SALIENCY=wrapper bash Rationale_Analysis/commands/saliency_script.sh;
SALIENCY=wrapper RATIONALE=top_k RATIONALE_EXP_NAME=direct bash Rationale_Analysis/commands/rationale_extractor_script.sh;
SALIENCY=wrapper RATIONALE=top_k RATIONALE_EXP_NAME=direct \
sbatch Cluster_scripts/multi_gpu_sbatch.sh bash Rationale_Analysis/commands/model_b_train_script.sh;