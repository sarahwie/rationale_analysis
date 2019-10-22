{
  dataset_reader : {
    type : "rationale_reader",
    tokenizer: {
       word_splitter: "spacy"
    },
    token_indexers : {
      tokens: {
        type: "single_id",
        lowercase_tokens: true
      },
    },
  },
  validation_dataset_reader: {
    type : "rationale_reader",
    tokenizer: {
       word_splitter: "spacy"
    },
    token_indexers : {
      tokens: {
        type: "single_id",
        lowercase_tokens: true
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
            tokens: {
                type: "embedding",
                embedding_dim: 300,
                pretrained_file: "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
                trainable: true
            },
        },
        seq2seq_encoder : {
            type: 'lstm',
            input_size: 300,
            hidden_size: 128,
            num_layers: 1,
            bidirectional: true
        },
        dropout: 0.2,
        feedforward_encoder:{
            input_dim: 256,
            num_layers: 1,
            hidden_dims: [128],
            activations: ['relu'],
            dropout: 0.2
        },
    },
    encoder : {
        type: "encoder_rationale_model",
        text_field_embedder: {
            tokens: {
                type: "embedding",
                embedding_dim: 300,
                pretrained_file: "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
                trainable: true
            },
        },
        seq2seq_encoder : {
            type: 'lstm',
            input_size: 300,
            hidden_size: 128,
            num_layers: 1,
            bidirectional: true
        },
        dropout: 0.2,
        attention: {
            type: 'additive',
            vector_dim: 256,
            matrix_dim: 256,
        },
        feedforward_encoder:{
            input_dim: 256,
            num_layers: 1,
            hidden_dims: [128],
            activations: ['relu'],
            dropout: 0.2
        },
    },
    samples: 1
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
    validation_metric: "+accuracy",
    cuda_device: std.extVar("CUDA_DEVICE"),
    optimizer: {
      type: "adam",
      lr: 1e-3
    }
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}
