# CLASSIFIER=bert_encoder_generator bash Rationale_Analysis/commands/model_a_train_script.sh

for rationale in top_k max_length;
        do
        SALIENCY=wrapper \
        RATIONALE=$rationale \
        RATIONALE_EXP_NAME=second_cut_point \
        bash Rationale_Analysis/commands/rationale_extractor_script.sh;

        SALIENCY=wrapper \
        RATIONALE=$rationale \
        RATIONALE_EXP_NAME=second_cut_point \
        sbatch Cluster_scripts/multi_gpu_sbatch.sh bash Rationale_Analysis/commands/model_b_train_script.sh;
        done;
    done;
