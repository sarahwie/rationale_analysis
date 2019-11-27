export CONFIG_FILE=Rationale_Analysis/training_config/rationale_generators/${RATIONALE_CLASSIFIER:?"Set classifier"}.jsonnet
export CUDA_DEVICE=${CUDA_DEVICE:?"set cuda device"}

export TRAIN_DATA_PATH=${DATA_BASE_PATH:?"set data base path"}/train.jsonl
export DEV_DATA_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_DATA_PATH=$DATA_BASE_PATH/test.jsonl

export default_output_base_path=${OUTPUT_DIR:-outputs}/${RATIONALE_CLASSIFIER}/${DATASET_NAME}/${EXP_NAME}
export OUTPUT_BASE_PATH=${OUTPUT_BASE_PATH:-$default_output_base_path}

export SEED=${RANDOM_SEED:-100}
export EPOCHS=${EPOCHS:-40}

function predict {
    allennlp predict \
    --output-file $1 \
    --batch-size ${BSIZE} \
    --use-dataset-reader \
    --dataset-reader-choice validation \
    --predictor rationale_predictor \
    --include-package Rationale_Analysis \
    --silent --cuda-device ${CUDA_DEVICE:?"set cuda device"} \
    $OUTPUT_BASE_PATH/model.tar.gz $2
}

predict $OUTPUT_BASE_PATH/train.jsonl $TRAIN_DATA_PATH
predict $OUTPUT_BASE_PATH/dev.jsonl $DEV_DATA_PATH
predict $OUTPUT_BASE_PATH/test.jsonl $TEST_DATA_PATH