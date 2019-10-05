if [ $# -eq  0 ]
  then
    echo "No argument supplied for dataset"
    exit 1
fi

export DATASET_NAME=$1
export DATA_BASE_PATH=Datasets/$1/data
export CUDA_DEVICE=0 

scorer=integrated_gradient_rationales

# . ./Rationale_Analysis/commands/bert_train_script.sh $1_base_bert_wrapper_min
. ./Rationale_Analysis/commands/bert_predict_script.sh $1_base_bert_wrapper_min ${scorer}

export DATA_BASE_PATH=${OUTPUT_BASE_PATH}/${scorer}
. ./Rationale_Analysis/commands/bert_train_script.sh $1_base_bert_wrapper_min/model_b_${scorer}
