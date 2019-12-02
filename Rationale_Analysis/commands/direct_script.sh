for saliency in wrapper simple_gradient;
    do 
    SALIENCY=$saliency bash Rationale_Analysis/commands/saliency_script.sh;
    for rationale in top_k max_length;
        do
        SALIENCY=$saliency RATIONALE=$rationale RATIONALE_EXP_NAME=direct bash Rationale_Analysis/commands/rationale_extractor_script.sh;

        SALIENCY=$saliency RATIONALE=$rationale RATIONALE_EXP_NAME=direct sbatch --time=$( t2sd ) Cluster_scripts/multi_gpu_sbatch.sh bash Rationale_Analysis/commands/model_b_train_script.sh;
        done;
    done;