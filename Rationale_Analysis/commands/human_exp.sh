KEEP_PROB=1.0 CLASSIFIER=bert_encoder_generator_human \
bash Rationale_Analysis/commands/model_a_train_script.sh;

CLASSIFIER=bert_encoder_generator_human \
RATIONALE=top_k \
RATIONALE_EXP_NAME=direct \
bash Rationale_Analysis/commands/model_a_rationale_extractor.sh;

# CLASSIFIER=bert_classification \
# EXP_NAME=direct/RANDOM_SEED=$RANDOM_SEED \
# bash Rationale_Analysis/commands/rationale_train_and_predict_script_human.sh