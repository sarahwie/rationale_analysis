for rationale in top_k max_length;
    do
    CLASSIFIER=bert_encoder_generator \
    RATIONALE=$rationale \
    RATIONALE_EXP_NAME=direct \
    bash Rationale_Analysis/commands/model_a_rationale_extractor.sh;
    done;