# CLASSIFIER=bert_encoder_generator sbatch Cluster_scripts/multi_gpu_sbatch.sh bash Rationale_Analysis/commands/model_a_train_script.sh;
for rationale in top_k;
    do
    CLASSIFIER=bert_encoder_generator \
    RATIONALE=$rationale \
    RATIONALE_EXP_NAME=direct \
    bash Rationale_Analysis/commands/model_a_rationale_extractor.sh;
    done;
# CLASSIFIER=bert_classification bash Rationale_Analysis/commands/model_a_train_script.sh;
# CLASSIFIER=bert_classification sbatch Cluster_scripts/multi_gpu_sbatch.sh bash Rationale_Analysis/hack_commands/top_k_wrapper_direct.sh;