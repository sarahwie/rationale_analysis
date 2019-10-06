export CONFIG_FILE=Rationale_Analysis/training_config/classifiers/bert_classification.jsonnet
export CUDA_DEVICE=$CUDA_DEVICE

SEED=10034
export SEED=$SEED

export TRAIN_DATA_PATH=$DATA_BASE_PATH/train.jsonl
export DEV_DATA_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_DATA_PATH=$DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/bert_classification/$DATASET_NAME/$EXP_NAME}

allennlp train -s $OUTPUT_BASE_PATH --include-package Rationale_Analysis --force $CONFIG_FILE