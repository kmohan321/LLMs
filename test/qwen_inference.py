import torch
from Qwen.model import QWEN3
from Qwen_moe.model import QWEN3_MOE
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# dense model config
config = {
  'hidden_dim': 1024,
  'head_dim': 128,
  'vocab_size': 151936,
  'num_blocks': 28,
  'intermediate_dim': 3072,
  'qk_norm': True,
  'causal': True,
  'max_seq_len': 40960,
  'num_kv_heads': 8,
  'num_q_heads': 16,
  'eps':1e-5
}

# moe model config
# config = {
#   'hidden_dim': 2048,
#   'head_dim': 128,
#   'vocab_size': 151936,
#   'num_blocks': 10,
#   'intermediate_dim': 768,
#   'qk_norm': True,
#   'causal': True,
#   'max_seq_len': 262144,
#   'num_kv_heads': 8,
#   'num_q_heads': 32,
#   'num_experts': 12,
#   'num_exp_tokens': 8 #num_experts for each token
# }

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hf_model_name = "Qwen/Qwen3-0.6B"

def load_weights(config, device, hf_model_name):
    
    model = QWEN3(config).to(device)

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
        
        mapping[f'blocks.{i}.attention.q_norm.weight'] = f'model.layers.{i}.self_attn.q_norm.weight'
        mapping[f'blocks.{i}.attention.k_norm.weight'] = f'model.layers.{i}.self_attn.k_norm.weight'
        
        if config['num_experts'] > 0:
            mapping[f'blocks.{i}.ffn.expert_layer.router.weight'] = f'model.layers.{i}.mlp.gate.weight'
            
            for j in config["num_experts"]:
                prefix = f'blocks.{i}.ffn.expert_layer.{j}'
                hf_prefix = f'model.layers.{i}.mlp.experts.{j}'
                mapping[f'{prefix}.gated.weight'] = f'{hf_prefix}.gate_proj.weight'
                mapping[f'{prefix}.w1.weight'] = f'{hf_prefix}.mlp.up_proj.weight'
                mapping[f'{prefix}.w2.weight'] = f'{hf_prefix}.mlp.down_proj.weight'    
                
        else:
            mapping[f'blocks.{i}.ffn.w1.weight'] = f'model.layers.{i}.mlp.gate_proj.weight'
            mapping[f'blocks.{i}.ffn.w2.weight'] = f'model.layers.{i}.mlp.down_proj.weight'
            mapping[f'blocks.{i}.ffn.w3.weight'] = f'model.layers.{i}.mlp.up_proj.weight'
            mapping[f'blocks.{i}.ffn_norm.weight'] = f'model.layers.{i}.post_attention_layernorm.weight'
        
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
    
    think_eos_id = tokenizer.convert_tokens_to_ids("</think>")
    eos_token_id = tokenizer.eos_token_id
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids = model_inputs.input_ids

    print(f"Generating from prompt...")
    print("-" * 40)

    generated_ids = input_ids.clone() # Start with the prompt
    past_kv = None
    generated_text_so_far = ""
    
    with torch.no_grad():
        for _ in range(max_new_tokens):

            if past_kv is None:
                out = model(generated_ids, past_kv) #prefilling the kv_cache
            else:
                out = model(generated_ids[:,-1:], past_kv) 
            
            next_token_logits, present_kv = out[0][:, -1, :], out[1]
            past_kv = present_kv
            
            if temperature > 0:
                logits_scaled = next_token_logits / temperature
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
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
             
            full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            new_text = full_text[len(generated_text_so_far):]
            
            print(new_text, end='', flush=True)
            generated_text_so_far = full_text

            if next_token.item() == eos_token_id:
                break

    prompt_len = len(model_inputs.input_ids[0])
    output_ids = generated_ids[0][prompt_len:].tolist() 

    index = len(output_ids) - output_ids[::-1].index(think_eos_id)

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("\n" + "="*30)
    # print("Thinking Content:")
    # print("="*30)
    # print(thinking_content)

    print("\n" + "="*30)
    print("Final Content:")
    print("="*30)
    print(content)
        

model = load_weights(config, device, hf_model_name)

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

prompt  = "Write a summary about the claude golden gate experiment"

generated_text = generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=1500, 
    temperature=0.7
)
