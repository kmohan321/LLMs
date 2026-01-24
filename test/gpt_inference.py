import torch
from GPT2.gpt_model import GPT
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

config = {
    "vocab_size": 50257,
    "hidden_dim": 1024,
    "blocks": 24,
    "heads": 16,
    "context_length": 1024,
    "mlp_multiplier": 4,
    "eps": 1e-5, 
    "drop": 0.1,
    "bias": True
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hf_model_name = "gpt2-medium"

def load_weights(config, device, hf_model_name):
    
    model = GPT(config).to(device)

    print("Downloading official weights...")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    hf_model = hf_model.to(device)
    hf_sd = hf_model.state_dict()

    mapping = {}
    mapping['token_embeddings.weight'] = 'transformer.wte.weight'
    mapping['pos_embeddings.weight'] = 'transformer.wpe.weight'
    
    mapping['model_norm.weight'] = 'transformer.ln_f.weight'
    mapping['model_norm.bias']   = 'transformer.ln_f.bias'
    mapping['final_layer.weight'] = 'lm_head.weight'

    for i in range(config['blocks']):
        
        mapping[f'blocks.{i}.attention.wqkv.weight'] = f'transformer.h.{i}.attn.c_attn.weight'
        mapping[f'blocks.{i}.attention.wqkv.bias']   = f'transformer.h.{i}.attn.c_attn.bias'
        mapping[f'blocks.{i}.attention.wo.weight'] = f'transformer.h.{i}.attn.c_proj.weight'
        mapping[f'blocks.{i}.attention.wo.bias']   = f'transformer.h.{i}.attn.c_proj.bias'

        mapping[f'blocks.{i}.attention_norm.weight'] = f'transformer.h.{i}.ln_1.weight'
        mapping[f'blocks.{i}.attention_norm.bias']   = f'transformer.h.{i}.ln_1.bias'
  
        mapping[f'blocks.{i}.ffn.w1.weight']   = f'transformer.h.{i}.mlp.c_fc.weight'
        mapping[f'blocks.{i}.ffn.w1.bias']     = f'transformer.h.{i}.mlp.c_fc.bias'
        mapping[f'blocks.{i}.ffn.w2.weight'] = f'transformer.h.{i}.mlp.c_proj.weight'
        mapping[f'blocks.{i}.ffn.w2.bias']   = f'transformer.h.{i}.mlp.c_proj.bias'
        
        mapping[f'blocks.{i}.ffn_norm.weight'] = f'transformer.h.{i}.ln_2.weight'
        mapping[f'blocks.{i}.ffn_norm.bias']   = f'transformer.h.{i}.ln_2.bias'

    my_state_dict = model.state_dict()
    keys_to_load = {}

    transpose_layers = [
    "attn.c_attn.weight", 
    "attn.c_proj.weight", 
    "mlp.c_fc.weight", 
    "mlp.c_proj.weight"
      ]
    
    print("Mapping weights...")
    for my_key, hf_key in mapping.items():
        if hf_key in hf_sd:
          param = hf_sd[hf_key]
          
          if any(x in hf_key for x in transpose_layers):
            param = param.t()
            
          if my_state_dict[my_key].shape == param.shape:
              keys_to_load[my_key] = param
              
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
        
        if tokens.shape[1] >= 1024:
            tokens = tokens[:, -1024:]

        with torch.no_grad():
          if past_kv is None:
            logits, present_kv = model(tokens, past_kv) #prefilling
          else:
            logits, present_kv = model(tokens[:, -1:], past_kv)
            
        past_kv = present_kv
        last_token_logits = logits[:, -1, :]
        
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
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

prompt  = """who are you?"""

generated_text = generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=500, 
    temperature=1,
    top_k=0,
    device=device
)

# print("\n" + "="*20 + " RESULT " + "="*20)
# print(generated_text)
