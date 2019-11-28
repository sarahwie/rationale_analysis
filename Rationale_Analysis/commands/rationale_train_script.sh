export CONFIG_FILE=Rationale_Analysis/training_config/rationale_generators/${RATIONALE_CLASSIFIER:?"Set classifier"}.jsonnet
export CUDA_DEVICE=${CUDA_DEVICE:?"set cuda device"}

export TRAIN_DATA_PATH=${DATA_BASE_PATH:?"set data base path"}/train.jsonl
export DEV_DATA_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_DATA_PATH=$DATA_BASE_PATH/test.jsonl

export default_output_base_path=${OUTPUT_DIR:-outputs}/${RATIONALE_CLASSIFIER}/${DATASET_NAME}/${EXP_NAME}
export OUTPUT_BASE_PATH=${OUTPUT_BASE_PATH:-$default_output_base_path}

export SEED=${RANDOM_SEED:-100}
export EPOCHS=${EPOCHS:-40}


if [[ -f "${OUTPUT_BASE_PATH}/metrics.json" ]]; then
    echo "${OUTPUT_BASE_PATH}/metrics.json exists ... . Not running Training ";
else 
    echo "${OUTPUT_BASE_PATH}/metrics.json exists ... . TRAINING ";
    # allennlp train -s $OUTPUT_BASE_PATH --include-package Rationale_Analysis --force $CONFIG_FILE
fi;