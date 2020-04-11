for search in {0..19};
    do
    EXP_NAME=mu_lambda_search/search_${search} sbatch Cluster_scripts/gpu_sbatch.sh bash Rationale_Analysis/commands/direct_lei_script.sh;
    done;