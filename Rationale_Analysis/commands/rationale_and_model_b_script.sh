export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${CLASSIFIER:?"Set classifier"}/${DATASET_NAME:?"Set dataset name"}/${EXP_NAME:?"Set Exp name"}

export TRAIN_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY:?"Set Saliency scorer"}_saliency/train.jsonl
export DEV_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/dev.jsonl
export TEST_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/test.jsonl

export RATIONALE_CONFIG_FILE=Rationale_Analysis/training_config/rationale_extractors/${RATIONALE:?"Set Rationale Extractor"}.jsonnet
export RATIONALE_FOLDER_NAME=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/${RATIONALE}_rationale/${RATIONALE_EXP_NAME:?"Set Rationale Extractor experiment name. May use hyperparameter settings for naming"}

mkdir -p $RATIONALE_FOLDER_NAME

function rationale {
    python -m Rationale_Analysis.commands.allennlp_runs rationale \
    --output-file $1 \
    --batch-size ${BSIZE} \
    --use-dataset-reader \
    --dataset-reader-choice validation \
    --predictor rationale_predictor \
    --include-package Rationale_Analysis \
    --silent --cuda-device ${CUDA_DEVICE:?"set cuda device"} \
    $RATIONALE_CONFIG_FILE $2
}

rationale $RATIONALE_FOLDER_NAME/train.jsonl $TRAIN_DATA_PATH
rationale $RATIONALE_FOLDER_NAME/dev.jsonl $DEV_DATA_PATH
rationale $RATIONALE_FOLDER_NAME/test.jsonl $TEST_DATA_PATH

DATA_BASE_PATH=$RATIONALE_FOLDER_NAME \
EXP_NAME=${EXP_NAME}/${SALIENCY}_saliency/${RATIONALE}_rationale/${RATIONALE_EXP_NAME}/model_b \
bash Rationale_Analysis/commands/model_train_script.sh