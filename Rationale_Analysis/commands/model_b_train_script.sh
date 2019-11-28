export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${CLASSIFIER:?"Set classifier"}/${DATASET_NAME:?"Set dataset name"}/${EXP_NAME:?"Set Exp name"}

export RATIONALE_CONFIG_FILE=Rationale_Analysis/training_config/rationale_extractors/${RATIONALE:?"Set Rationale Extractor"}.jsonnet
export RATIONALE_FOLDER_NAME=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/${RATIONALE}_rationale/${RATIONALE_EXP_NAME:?"Set Rationale Extractor experiment name. May use hyperparameter settings for naming"}

export KEEP_PROB=${KEEP_PROB:-1.0}
export BSIZE=${BSIZE:-50}

DATA_BASE_PATH=$RATIONALE_FOLDER_NAME \
EXP_NAME=${EXP_NAME}/${SALIENCY}_saliency/${RATIONALE}_rationale/${RATIONALE_EXP_NAME}/model_b \
bash Rationale_Analysis/commands/model_train_script.sh