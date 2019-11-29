export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${CLASSIFIER:?"Set classifier"}/${DATASET_NAME:?"Set dataset name"}/${EXP_NAME:?"Set Exp name"}

export TRAIN_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY:?"Set Saliency scorer"}_saliency/train.jsonl
export DEV_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/dev.jsonl
export TEST_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/test.jsonl

export RATIONALE_CONFIG_FILE=Rationale_Analysis/training_config/rationale_extractors/${RATIONALE:?"Set Rationale Extractor"}.jsonnet
export RATIONALE_FOLDER_NAME=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/${RATIONALE}_rationale/${RATIONALE_EXP_NAME:?"Set Rationale Extractor experiment name. May use hyperparameter settings for naming"}

DATA_BASE_PATH=$RATIONALE_FOLDER_NAME \
OUTPUT_BASE_PATH=${OUTPUT_BASE_PATH}/${SALIENCY}_saliency/${RATIONALE}_rationale/${RATIONALE_EXP_NAME}/${RATIONALE_CLASSIFIER}/${RC_EXP_NAME} \
bash Rationale_Analysis/commands/rationale_train_script.sh

DATA_BASE_PATH=$RATIONALE_FOLDER_NAME \
OUTPUT_BASE_PATH=${OUTPUT_BASE_PATH}/${SALIENCY}_saliency/${RATIONALE}_rationale/${RATIONALE_EXP_NAME}/${RATIONALE_CLASSIFIER}/${RC_EXP_NAME} \
bash Rationale_Analysis/commands/rationale_predict_script.sh
