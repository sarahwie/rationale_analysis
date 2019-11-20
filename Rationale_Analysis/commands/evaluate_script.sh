export DATA_BASE_PATH=${DATASETS_FOLDER:-Datasets}/${DATASET_NAME:?"Set dataset name"}/data

export TRAIN_DATA_PATH=${DATA_BASE_PATH}/train.jsonl
export DEV_DATA_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_DATA_PATH=$DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${CLASSIFIER:?"Set classifier"}/${DATASET_NAME:?"Set dataset name"}/${EXP_NAME:?"Set Exp name"}

function saliency {
    allennlp evaluate \
    --include-package Rationale_Analysis \
    --cuda-device ${CUDA_DEVICE:?"set cuda device"} \
    $OUTPUT_BASE_PATH/model.tar.gz $2
}

saliency $SALIENCY_FOLDER_NAME/test.jsonl $TEST_DATA_PATH
saliency $SALIENCY_FOLDER_NAME/train.jsonl $TRAIN_DATA_PATH
saliency $SALIENCY_FOLDER_NAME/dev.jsonl $DEV_DATA_PATH
