{
    dataset_reader : {
        type : "saliency_reader",
    },
    validation_dataset_reader : {
        type : "saliency_reader",
    },
    model : {
        type : 'min_attention',
        min_attention_score: std.parseFloat(std.extVar('MIN_ATTN_SCORE'))
    },
}