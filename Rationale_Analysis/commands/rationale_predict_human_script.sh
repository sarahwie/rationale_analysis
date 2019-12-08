export CONFIG_FILE=Rationale_Analysis/training_config/rationale_generators/bert_generator_human.jsonnet
export CUDA_DEVICE=${CUDA_DEVICE:?"set cuda device"}

export TRAIN_DATA_PATH="${DATASET_FOLDER}/${DATASET_NAME}/data/train.jsonl;${OUTPUT_BASE_PATH:?"set data base path"}/train.jsonl"
export DEV_DATA_PATH="${DATASET_FOLDER}/${DATASET_NAME}/data/dev.jsonl;${OUTPUT_BASE_PATH:?"set data base path"}/dev.jsonl"
export TEST_DATA_PATH="${DATASET_FOLDER}/${DATASET_NAME}/data/test.jsonl;${OUTPUT_BASE_PATH:?"set data base path"}/test.jsonl"

export OUTPUT_BASE_PATH=${OUTPUT_BASE_PATH:-outputs}/bert_generator_human/no_crf_${HUMAN_PROB}

export SEED=${RANDOM_SEED:-100}
export EPOCHS=${EPOCHS:-40}

function predict {
    if [ -f "$1" ]; then
        echo "$1 exists ... Not running Rationale Train ";
    else 
        echo "$1 do not exist RUNNING RATIONALE Train ";
        allennlp predict \
        --output-file $1 \
        --batch-size ${BSIZE} \
        --use-dataset-reader \
        --dataset-reader-choice validation \
        --predictor rationale_predictor \
        --include-package Rationale_Analysis \
        --silent --cuda-device ${CUDA_DEVICE:?"set cuda device"} \
        $OUTPUT_BASE_PATH/model.tar.gz $2;
    fi;
}

predict $OUTPUT_BASE_PATH/train.jsonl $TRAIN_DATA_PATH
predict $OUTPUT_BASE_PATH/dev.jsonl $DEV_DATA_PATH
predict $OUTPUT_BASE_PATH/test.jsonl $TEST_DATA_PATH