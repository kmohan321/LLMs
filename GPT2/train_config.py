config = {
    "file_path": "replace with the data location",
    "context_length": 128,
    "stride": 128,
    "batch_size": 32,
    "train_ratio": 0.85,
    "vocab_size": 50257,
    
    "hidden_dim": 256,
    "heads": 4,
    "blocks": 10,
    "dropout": 0.2,
    "bias": False,

    "epochs": 200,
    "learning_rate": 1e-4,
    "weight_decay": 0.1,
    
    "eval_prompt": "Hardwork and dedication",
    "max_gen_tokens": 15
}