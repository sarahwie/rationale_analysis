########################################################
# Predict saliency on given dataset and trained model
# Parameters required for this script
# CUDA_DEVICE , RATIONALE , DATASET_NAME , EXP_NAME , SALIENCY 

export CUDA_DEVICE=$CUDA_DEVICE

export CONFIG_FILE=Rationale_Analysis/training_config/bert_classification.jsonnet
export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/bert_classification/$DATASET_NAME/$EXP_NAME}

export TRAIN_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/train.jsonl
export DEV_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/dev.jsonl
export TEST_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/test.jsonl

export RATIONALE_CONFIG_FILE=Rationale_Analysis/training_config/rationale_extractors/${RATIONALE}.jsonnet
export RATIONALE_FOLDER_NAME=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/${RATIONALE}_rationale

mkdir -p $RATIONALE_FOLDER_NAME

function rationale {
    python -m Rationale_Analysis.commands.allennlp_runs rationale \
    --output-file $1 \
    --batch-size 50 \
    --use-dataset-reader \
    --dataset-reader-choice validation \
    --predictor rationale_predictor \
    --include-package Rationale_Analysis \
    --silent --cuda-device 0 \
    $RATIONALE_CONFIG_FILE $2
}

rationale $RATIONALE_FOLDER_NAME/train.jsonl $TRAIN_DATA_PATH
rationale $RATIONALE_FOLDER_NAME/dev.jsonl $DEV_DATA_PATH
rationale $RATIONALE_FOLDER_NAME/test.jsonl $TEST_DATA_PATH

DATASET_NAME=$DATASET_NAME \
DATA_BASE_PATH=$RATIONALE_FOLDER_NAME \
EXP_NAME=${EXP_NAME}/${SALIENCY}_saliency/${RATIONALE}_rationale/model_b \
bash Rationale_Analysis/commands/bert_train_script.sh