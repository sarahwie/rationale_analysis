{
    dataset_reader : {
        type : "saliency_reader",
    },
    validation_dataset_reader : {
        type : "saliency_reader",
    },
    model : {
        type : 'top_k',
        max_length_ratio: std.parseInt(std.extVar('MAX_LENGTH_PERCENT')) / 100
    },
}