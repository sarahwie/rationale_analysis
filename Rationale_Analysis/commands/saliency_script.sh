export DATA_BASE_PATH = ${DATASETS_FOLDER:-Datasets}/${DATASET_NAME:?"Set dataset name"}/data

export TRAIN_DATA_PATH=${DATA_BASE_PATH}/train.jsonl
export DEV_DATA_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_DATA_PATH=$DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${CLASSIFIER:?"Set classifier"}/${DATASET_NAME:?"Set dataset name"}/${EXP_NAME:?"Set Exp name"}

export SALIENCY_CONFIG_FILE=Rationale_Analysis/training_config/saliency_scorers/${SALIENCY:?"Set Saliency scorer"}.jsonnet
export SALIENCY_FOLDER_NAME=$OUTPUT_BASE_PATH/${SALIENCY}_saliency

mkdir -p $SALIENCY_FOLDER_NAME

function saliency {
    python -m Rationale_Analysis.commands.allennlp_runs saliency \
    --output-file $1 \
    --batch-size 1 \
    --use-dataset-reader \
    --dataset-reader-choice validation \
    --predictor rationale_predictor \
    --include-package Rationale_Analysis \
    --silent --cuda-device ${CUDA_DEVICE:?"set cuda device"} \
    $OUTPUT_BASE_PATH/model.tar.gz $SALIENCY_CONFIG_FILE $2
}

saliency $SALIENCY_FOLDER_NAME/train.jsonl $TRAIN_DATA_PATH
saliency $SALIENCY_FOLDER_NAME/dev.jsonl $DEV_DATA_PATH
saliency $SALIENCY_FOLDER_NAME/test.jsonl $TEST_DATA_PATH