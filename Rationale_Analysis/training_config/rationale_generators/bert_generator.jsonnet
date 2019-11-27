{
  dataset_reader : {
    type : "rationale_reader",
    add_rationale: true,
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
    add_rationale: true,
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
    type: "bert_middle_model",
    bert_model: 'bert-base-uncased',
    requires_grad: 'pooler,11,10,9',
    dropout : 0.2,
    use_crf: false
  },
  iterator: {
    type: "bucket",
    sorting_keys: [["document", "num_tokens"]],
    batch_size : std.extVar('BSIZE')
  },
  trainer: {
    num_epochs: std.extVar('EPOCHS'),
    patience: 20,
    grad_norm: 5.0,
    validation_metric: "+f1",
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
