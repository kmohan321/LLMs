import torch
from Llama.model import LLama_Basic
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

config = {
    "vocab_size": 32000,
    "hidden_dims": 2048,
    "intermediate_size": 5632,
    "num_heads_q": 32,
    "num_heads_kv": 4,
    "num_blocks": 22,
    "seq_length": 2048,
    "eps": 1e-5
    }

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hf_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_weights(config, device, hf_model_name):
    
    model = LLama_Basic(config).to(device)

    print("Downloading official weights...")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    hf_model = hf_model.to(device)
    hf_sd = hf_model.state_dict()

    mapping = {}
    mapping['embedding.weight'] = 'model.embed_tokens.weight'
    mapping['model_norm.weight'] = 'model.norm.weight'
    mapping['final_layer.weight'] = 'lm_head.weight'


    for i in range(config['num_blocks']):
        
        mapping[f'blocks.{i}.attention.wq.weight'] = f'model.layers.{i}.self_attn.q_proj.weight'
        mapping[f'blocks.{i}.attention.wk.weight'] = f'model.layers.{i}.self_attn.k_proj.weight'
        mapping[f'blocks.{i}.attention.wv.weight'] = f'model.layers.{i}.self_attn.v_proj.weight'
        mapping[f'blocks.{i}.attention.wo.weight'] = f'model.layers.{i}.self_attn.o_proj.weight'
        mapping[f'blocks.{i}.attention_norm.weight'] = f'model.layers.{i}.input_layernorm.weight'
        
        mapping[f'blocks.{i}.ffn.w1.weight'] = f'model.layers.{i}.mlp.gate_proj.weight'
        mapping[f'blocks.{i}.ffn.w2.weight'] = f'model.layers.{i}.mlp.down_proj.weight'
        mapping[f'blocks.{i}.ffn.w3.weight'] = f'model.layers.{i}.mlp.up_proj.weight'
        mapping[f'blocks.{i}.ffn_norm.weight'] = f'model.layers.{i}.post_attention_layernorm.weight'

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


def generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=100, 
    temperature=0.7, 
    top_k=50, 
    device=device
):
    model.eval()
    model.to(device)
    
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"Generating from prompt...")
    print("-" * 40)
  
    generated_text_so_far = ""
    past_kv = None
    for _ in range(max_new_tokens):
        
        if tokens.shape[1] >= 2048:
            tokens = tokens[:, -2048:]

        with torch.no_grad():
            if past_kv is None:
                logits, present_kv = model(tokens, past_kv) #prefilling 
            else:
                logits, present_kv = model(tokens[:, -1:], past_kv)
        
        last_token_logits = logits[:, -1]
        past_kv = present_kv
        
        if temperature > 0:
            logits_scaled = last_token_logits / temperature
            probs = torch.softmax(logits_scaled, dim=-1)
            
            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                # Sample from the top K
                next_token_index = torch.multinomial(top_k_probs, num_samples=1)
                # Map back to real vocab index
                next_token = torch.gather(top_k_indices, -1, next_token_index)
            else:
                # Sample from full distribution
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)

        tokens = torch.cat([tokens, next_token], dim=1)
        full_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        new_text = full_text[len(generated_text_so_far):]
        
        print(new_text, end='', flush=True)
        generated_text_so_far = full_text
        
        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated_text_so_far

model = load_weights(config, device, hf_model_name)

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

prompt  = "<|user|>\nWrite a Python function to calculate factorial.\n<|assistant|>\n"

generated_text = generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=300, 
    temperature=0.7
)

# print("\n" + "="*20 + " RESULT " + "="*20)
# print(generated_text)
