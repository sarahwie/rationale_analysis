for saliency in wrapper simple_gradient;
    do 
    SALIENCY=$saliency bash Rationale_Analysis/commands/saliency_script.sh;
    for rationale in top_k max_length;
        do
        SALIENCY=$saliency RATIONALE=$rationale RATIONALE_EXP_NAME=direct \
        RATIONALE_CLASSIFIER=bert_generator_saliency RC_EXP_NAME=direct USE_CRF=true \
        sbatch Cluster_scripts/multi_gpu_sbatch.sh bash Rationale_Analysis/commands/rationale_train_and_predict_script.sh;
       done;
    done;