{
  dataset_reader : {
    type : "rationale_reader",
    tokenizer: {
       word_splitter: "bert-basic"
    },
    token_indexers : {
      bert : {
        type : "bert-pretrained-simple",
        pretrained_model : "bert-base-uncased",
        use_starting_offsets: true,
        do_lowercase : true,
        truncate_long_sequences: false
      },
    },
    keep_prob: std.extVar('KEEP_PROB')
  },
  validation_dataset_reader: {
    type : "rationale_reader",
    tokenizer: {
       word_splitter: "bert-basic"
    },
    token_indexers : {
      bert : {
        type : "bert-pretrained-simple",
        pretrained_model : "bert-base-uncased",
        use_starting_offsets: true,
        do_lowercase : true,
        truncate_long_sequences: false
      },
    },
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: {
    type: "encoder_generator_rationale_model",
    generator: {
      type: "bert_generator_model",
      bert_model: 'bert-base-uncased',
      requires_grad: 'pooler,11,10,9',
      dropout : 0.2,
    },
    encoder : {
      type: "bert_rationale_model",
      bert_model: 'bert-base-uncased',
      requires_grad: 'pooler,11,10,9',
      dropout : 0.2,
    },
    samples: 1,
    reg_loss_lambda: std.extVar('LAMBDA'),
    reg_loss_mu: std.extVar('MU'),
    desired_length: std.extVar('MAX_LENGTH_RATIO')
  },
  iterator: {
    type: "bucket",
    sorting_keys: [["document", "num_tokens"]],
    batch_size : std.extVar('BSIZE')
  },
  trainer: {
    num_epochs: 40,
    patience: 20,
    grad_norm: 5.0,
    validation_metric: "+reg_accuracy",
    num_serialized_models_to_keep: 1,
    cuda_device: std.extVar("CUDA_DEVICE"),
    optimizer: {
      type: "adam",
      lr: 2e-5
    }
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}
