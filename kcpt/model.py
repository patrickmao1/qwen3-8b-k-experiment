def load_model(model_id, max_seq_length):
    """Load a (base or adapter) model in 4-bit for inference via unsloth."""
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tok
