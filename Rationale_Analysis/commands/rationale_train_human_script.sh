export CONFIG_FILE=Rationale_Analysis/training_config/rationale_generators/bert_generator_human.jsonnet
export CUDA_DEVICE=${CUDA_DEVICE:?"set cuda device"}
export USE_CRF=false

export TRAIN_DATA_PATH="${DATASETS_FOLDER}/${DATASET_NAME}/data/train.jsonl;${OUTPUT_BASE_PATH:?"set data base path"}/train.jsonl"
export DEV_DATA_PATH="${DATASETS_FOLDER}/${DATASET_NAME}/data/dev.jsonl;${OUTPUT_BASE_PATH:?"set data base path"}/dev.jsonl"
export TEST_DATA_PATH="${DATASETS_FOLDER}/${DATASET_NAME}/data/test.jsonl;${OUTPUT_BASE_PATH:?"set data base path"}/test.jsonl"

export OUTPUT_BASE_PATH=${OUTPUT_BASE_PATH:-outputs}/bert_generator_human/no_crf_${HUMAN_PROB}

export SEED=${RANDOM_SEED:-100}
export EPOCHS=${EPOCHS:-20}


if [[ -f "${OUTPUT_BASE_PATH}/metrics.json" ]]; then
    echo "${OUTPUT_BASE_PATH}/metrics.json exists ... . Not running Training ";
else 
    echo "${OUTPUT_BASE_PATH}/metrics.json exists ... . TRAINING ";
    allennlp train -s $OUTPUT_BASE_PATH --include-package Rationale_Analysis --force $CONFIG_FILE
fi;