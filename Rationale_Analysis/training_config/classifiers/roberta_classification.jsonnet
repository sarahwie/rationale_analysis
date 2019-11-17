{
  dataset_reader : {
    type : "rationale_reader",
    tokenizer: {
       word_splitter: "bert-basic"
    },
    token_indexers : {
      roberta : {
        type : "roberta-pretrained-simple",
        model_name : "roberta-base",
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
      roberta : {
        type : "roberta-pretrained-simple",
        model_name : "roberta-base",
      },
    },
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: {
    type: "roberta_rationale_model",
    roberta_model: 'roberta-base',
    requires_grad: 'pooler,11,10,9',
    dropout : 0.2,
  },
  iterator: {
    type: "bucket",
    sorting_keys: [["document", "num_tokens"]],
    batch_size : std.extVar('BSIZE')
  },
  trainer: {
    num_epochs: 40,
    patience: 20,
    grad_norm: 0.0,
    validation_metric: "+accuracy",
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
