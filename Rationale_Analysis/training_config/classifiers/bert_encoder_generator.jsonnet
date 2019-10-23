{
  dataset_reader : {
    type : "bert_rationale_reader",
    tokenizer: {
       word_splitter: "bert-basic"
    },
    token_indexers : {
      bert : {
        type : "bert-pretrained",
        pretrained_model : "bert-base-uncased",
        use_starting_offsets: true,
        do_lowercase : true,
      },
    },
  },
  validation_dataset_reader: {
    type : "bert_rationale_reader",
    tokenizer: {
       word_splitter: "bert-basic"
    },
    token_indexers : {
      bert : {
        type : "bert-pretrained",
        pretrained_model : "bert-base-uncased",
        use_starting_offsets: true,
        do_lowercase : true,
      },
    },
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: {
    type: "encoder_generator_rationale_model",
    generator: {
      type: "simple_generator_model",
      text_field_embedder: {
          allow_unmatched_keys: true,
          embedder_to_indexer_map: {
            bert: ["bert", "bert-offsets"],
          },
          token_embedders: {
            bert: {
              type: "bert-pretrained",
              pretrained_model: 'bert-base-uncased',
              requires_grad: '11'
            },
          },
      },
      seq2seq_encoder : {
          type: 'pass_through',
          input_dim: 768
      },
      dropout: 0.3,
      feedforward_encoder:{
          type: 'pass_through',
          input_dim: 768
      },
    },
    encoder : {
      type: "bert_rationale_model",
      bert_model: 'bert-base-uncased',
      requires_grad: '11',
      dropout : 0.3,
    },
    samples: 1,
    reg_loss_lambda: 0.1,
    desired_length: std.extVar('MAX_LENGTH_RATIO')
  },
  iterator: {
    type: "bucket",
    sorting_keys: [["document", "num_tokens"]],
    batch_size : 20
  },
  trainer: {
    num_epochs: 40,
    patience: 20,
    grad_norm: 10.0,
    validation_metric: "+reg_accuracy",
    num_serialized_models_to_keep: 1,
    cuda_device: std.extVar("CUDA_DEVICE"),
    optimizer: {
      type: "adam",
      lr: 2e-5
    }
  },
  evaluate_on_test: true
}
