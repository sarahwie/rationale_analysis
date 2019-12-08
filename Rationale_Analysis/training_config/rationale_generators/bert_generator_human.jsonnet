local is_movies = if std.findSubstr('movies', std.extVar('TRAIN_DATA_PATH')) == [] then false else true;

local bert_gen_model = {
  type: "bert_middle_model",
  bert_model: 'bert-base-uncased',
  requires_grad: 'pooler,11,10,9',
  dropout : 0.2,
  use_crf: std.extVar('USE_CRF'),
  pos_weight: std.extVar('MAX_LENGTH_RATIO')
};

local simple_gen_model = {
  type: "simple_middle_model",
  text_field_embedder: {
    allow_unmatched_keys: true,
    embedder_to_indexer_map: {
      bert: ["bert", "bert-offsets"],
    },
    token_embedders: {
      bert: {
        type: "bert-pretrained",
        pretrained_model: 'bert-base-uncased',
        requires_grad: '11,10,pooler',
      },
    },
  },
  seq2seq_encoder : {
    type: 'lstm',
    input_size: 768,
    hidden_size: 128,
    num_layers: 1,
    bidirectional: true
  },
  dropout: 0.2,
  feedforward_encoder:{
    type: 'pass_through',
    input_dim: 256
  },
  use_crf: std.extVar('USE_CRF'),
  pos_weight: std.extVar('MAX_LENGTH_RATIO')
};

local indexer = if is_movies then "bert-pretrained" else "bert-pretrained-simple";

{
  dataset_reader : {
    type : "human_reader",
    token_indexers : {
      bert : {
        type : indexer,
        pretrained_model : "bert-base-uncased",
        use_starting_offsets: true,
        do_lowercase : true,
        truncate_long_sequences: false
      },
    },
    human_prob: std.extVar('HUMAN_PROB')
  },
  validation_dataset_reader: {
    type : "human_reader",
    token_indexers : {
      bert : {
        type : indexer,
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
  model: if is_movies then simple_gen_model else bert_gen_model,
  iterator: {
    type: "bucket",
    sorting_keys: [["document", "num_tokens"]],
    batch_size : std.extVar('BSIZE')
  },
  trainer: {
    num_epochs: 20,
    patience: 10,
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
