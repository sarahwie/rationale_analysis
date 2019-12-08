export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${CLASSIFIER:?"Set classifier"}/${DATASET_NAME:?"Set dataset name"}/${EXP_NAME:?"Set Exp name"}

export TRAIN_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY:?"Set Saliency scorer"}_saliency/train.jsonl
export DEV_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/dev.jsonl
export TEST_DATA_PATH=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/test.jsonl

export RATIONALE=top_k
export RATIONALE_EXP_NAME=human

bash Rationale_Analysis/commands/rationale_extractor_script.sh

export RATIONALE_FOLDER_NAME=$OUTPUT_BASE_PATH/${SALIENCY}_saliency/${RATIONALE}_rationale/${RATIONALE_EXP_NAME:?"Set Rationale Extractor experiment name. May use hyperparameter settings for naming"}

OUTPUT_BASE_PATH=${RATIONALE_FOLDER_NAME} \
bash Rationale_Analysis/commands/rationale_train_human_script.sh

OUTPUT_BASE_PATH=${RATIONALE_FOLDER_NAME} \
bash Rationale_Analysis/commands/rationale_predict_human_script.sh

RATIONALE_CLASSIFIER=bert_generator_human \
RC_EXP_NAME=/no_crf_${HUMAN_PROB} \
sbatch Cluster_scripts/multi_gpu_sbatch.sh bash Rationale_Analysis/commands/rationale_train_model_b_train_script.sh;