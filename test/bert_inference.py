import torch
from transformers import BertModel, BertTokenizer
from Bert.model import BERT

config = {
  "vocab_size": 30522,
  "hidden_dim": 768,              
  "blocks": 12,            
  "heads": 12,          
  "ffn_multiplier": 4,                       
  "drop_hidden": 0.1,         
  "drop_mha": 0.1, 
  "max_seq_length": 512,   
  "type_vocab_size": 2,                        
  "eps": 1e-12,            
  "pad_token_id": 0                   
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hf_model_name = "bert-base-uncased"

def load_weights(config, device, hf_model_name):
    
    model = BERT(config).to(device)

    print("Downloading official weights...")
    hf_model = BertModel.from_pretrained(hf_model_name)
    hf_model = hf_model.to(device)
    hf_sd = hf_model.state_dict()
    # print(hf_sd.keys())

    mapping = {}
    mapping['embeddings.word_embeddings.weight'] = 'embeddings.word_embeddings.weight'
    mapping['embeddings.position_embeddings.weight'] = 'embeddings.position_embeddings.weight'
    mapping['embeddings.token_type_embeddings.weight'] = 'embeddings.token_type_embeddings.weight'
    mapping['embeddings.layer_norm.weight'] = 'embeddings.LayerNorm.weight'
    mapping['embeddings.layer_norm.bias'] = 'embeddings.LayerNorm.bias'
    mapping['pooler_layer.weight'] = 'pooler.dense.weight'
    mapping['pooler_layer.bias'] = 'pooler.dense.bias'

    for i in range(config['blocks']):
        
        mapping[f'blocks.{i}.attention.wq.weight'] = f'encoder.layer.{i}.attention.self.query.weight'
        mapping[f'blocks.{i}.attention.wq.bias'] = f'encoder.layer.{i}.attention.self.query.bias'
        mapping[f'blocks.{i}.attention.wk.weight'] = f'encoder.layer.{i}.attention.self.key.weight'
        mapping[f'blocks.{i}.attention.wk.bias'] = f'encoder.layer.{i}.attention.self.key.bias'
        mapping[f'blocks.{i}.attention.wv.weight'] = f'encoder.layer.{i}.attention.self.value.weight'
        mapping[f'blocks.{i}.attention.wv.bias'] = f'encoder.layer.{i}.attention.self.value.bias'
        mapping[f'blocks.{i}.attention.wo.weight'] = f'encoder.layer.{i}.attention.output.dense.weight'
        mapping[f'blocks.{i}.attention.wo.bias'] = f'encoder.layer.{i}.attention.output.dense.bias'
        mapping[f'blocks.{i}.norm_mha.weight'] = f'encoder.layer.{i}.attention.output.LayerNorm.weight'
        mapping[f'blocks.{i}.norm_mha.bias'] = f'encoder.layer.{i}.attention.output.LayerNorm.bias'
        
        mapping[f'blocks.{i}.ffn.w1.weight'] = f'encoder.layer.{i}.intermediate.dense.weight'
        mapping[f'blocks.{i}.ffn.w1.bias'] = f'encoder.layer.{i}.intermediate.dense.bias'
        mapping[f'blocks.{i}.ffn.w2.weight'] = f'encoder.layer.{i}.output.dense.weight'
        mapping[f'blocks.{i}.ffn.w2.bias'] = f'encoder.layer.{i}.output.dense.bias'
        mapping[f'blocks.{i}.norm_ffn.weight'] = f'encoder.layer.{i}.output.LayerNorm.weight'
        mapping[f'blocks.{i}.norm_ffn.bias'] = f'encoder.layer.{i}.output.LayerNorm.bias'

    my_state_dict = model.state_dict()
    keys_to_load = {}

    print("Mapping weights...")
    for my_key, hf_key in mapping.items():
        
        if hf_key in hf_sd:
            
            if my_state_dict[my_key].shape == hf_sd[hf_key].shape:
                keys_to_load[my_key] = hf_sd[hf_key]
            else:
                print(f"Shape Mismatch at {my_key}: Yours {my_state_dict[my_key].shape} vs HF {hf_sd[hf_key].shape}")
        else:
            print(f"Missing key in HF model: {hf_key}")

    model.load_state_dict(keys_to_load, strict=False)
    print("Weights loaded successfully!")
    del hf_model
    del hf_sd
    torch.cuda.empty_cache()
    return model
  
model = load_weights(config, device, hf_model_name)
model.eval()
hf_model = BertModel.from_pretrained(hf_model_name, device_map = device)
tokenizer = BertTokenizer.from_pretrained(hf_model_name)
hf_model.eval()

def compare_tensors(name, ref_tensor, my_tensor, atol=1e-5):
    """Helper to compare two tensors and print the result."""
    # Detach and move to cpu for comparison
    ref = ref_tensor.detach().cpu()
    my = my_tensor.detach().cpu()
    
    if not torch.allclose(ref, my, atol=atol):
        diff = (ref - my).abs().mean().item()
        print(f"❌ {name} Mismatch! Avg Diff: {diff:.6f}")
        return False
    print(f"✅ {name} Match!")
    return True


# ==========================================
# TEST 1: Single Sentence (No Segment Info)
# ==========================================
print("\n--- TEST 1: Single Sentence ---")
text_a = "Hello, I am testing my BERT implementation."

inputs = tokenizer(text_a, return_tensors="pt", padding="max_length", max_length=config['max_seq_length'])
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
print(f"Input Shape: {input_ids.shape}")

with torch.no_grad():
    hf_outputs = hf_model(input_ids=input_ids, attention_mask=attention_mask)
    hf_seq = hf_outputs.last_hidden_state
    hf_pool = hf_outputs.pooler_output

with torch.no_grad():
    my_seq, my_pool = model(input_ids) 

compare_tensors("Sequence Output", hf_seq, my_seq)
compare_tensors("Pooled Output  ", hf_pool, my_pool)


# ==========================================
# TEST 2: Sentence Pair (With Segment Info)
# ==========================================
print("\n--- TEST 2: Sentence Pair (Passing Segment Info) ---")
text_a = "How does the encoder work?"
text_b = "It uses self-attention mechanisms."

inputs_pair = tokenizer(text_a, text_b, return_tensors="pt", padding="max_length", max_length=config['max_seq_length'])
input_ids_pair = inputs_pair["input_ids"].to(device)
attention_mask_pair = inputs_pair["attention_mask"].to(device)
token_type_ids_pair = inputs_pair["token_type_ids"].to(device)

print(f"Input Shape: {input_ids_pair.shape}")
print(f"Token Types: {token_type_ids_pair[0, :15]}...")

with torch.no_grad():
    hf_outputs_2 = hf_model(
        input_ids=input_ids_pair, 
        attention_mask=attention_mask_pair, 
        token_type_ids=token_type_ids_pair
    )
    hf_seq_2 = hf_outputs_2.last_hidden_state
    hf_pool_2 = hf_outputs_2.pooler_output

with torch.no_grad():
    my_seq_2, my_pool_2 = model(input_ids_pair, token_type_ids_pair)

compare_tensors("Sequence Output", hf_seq_2, my_seq_2)
compare_tensors("Pooled Output  ", hf_pool_2, my_pool_2)
