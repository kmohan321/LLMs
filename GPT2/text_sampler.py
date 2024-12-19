import torch

def text_generater(model,tokenizer,eval_prompt,context_length,
                   max_tokens,temp_scale,device,top_k=None):
  '''
  Args:
    model (nn.Module): The language model.
    tokenizer: The tokenizer for encoding and decoding text.
    eval_prompt (str): Initial text prompt to start generation.
    context_length (int): Maximum number of tokens to consider as context.
    max_tokens (int): Number of tokens to generate.
    temp_scale (float): Temperature scaling factor for logits.
    device (str): Device to run the model ('cuda' or 'cpu').
    top_k (int, optional): If specified, limits sampling to top-k logits.
    
  '''

  token_ids = tokenizer.encode(eval_prompt)
  token_ids = torch.tensor(token_ids).unsqueeze(0)
  token_ids = token_ids.to(device)
  model.eval()
  
  for _ in range(max_tokens):
    input_ids = token_ids[:,-context_length:]
    with torch.no_grad():
      logits = model(input_ids)[:,-1,:]
      
      if top_k is not None:               
        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(
            logits < min_val,
            torch.tensor(float('-inf')).to(logits.device),
            logits)
      
      if temp_scale > 0:
        probs = torch.softmax(logits/temp_scale,dim=-1)
        next_id = torch.multinomial(probs,1)
      else:
        probs = torch.softmax(logits,dim=-1)
        next_id = torch.argmax(probs,dim=-1,keepdim=True)
        
      token_ids = torch.cat([token_ids,next_id],dim=-1)
  text = tokenizer.decode(token_ids.squeeze(0).tolist())
  return text