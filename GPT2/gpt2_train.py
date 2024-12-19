import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from gpt_model import GPT
from data import DATASET
from torch.amp.grad_scaler import GradScaler
import tiktoken as tk

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = tk.get_encoding('gpt2')

#change the file path for text , or adjust as per file type 
with open('./data/shakespeare.txt','r',encoding="utf-8") as f:
  text = f.read()
  print(len(text))


context_length = 128
stride = 128
batch_size = 32
epochs = 200
max_tokens = 15
train_ratio = 0.85
heads = 4
vocab_size = 50257
blocks = 10
hidden_dim = 256
eval_prompt = 'Hardwork and dedication' #change it 

split_idx = int(train_ratio * len(text))
train_data = text[:split_idx]
val_data = text[split_idx:]

train_dataset = DATASET(train_data,context_length,stride)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
val_dataset = DATASET(val_data,context_length,stride)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,drop_last=True)

model = GPT(hidden_dim,vocab_size,context_length,heads,blocks,drop=0.2,bias=False)
model = model.to(device)
scaler = GradScaler(device= device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr = 0.0001,weight_decay=0.1)
print(f"The number of parameters: {sum(p.numel() for p in model.parameters())}")


def text_generater(tokenizer,eval_prompt,context_length,
                   max_tokens):
  token_ids = tokenizer.encode(eval_prompt)
  token_ids = torch.tensor(token_ids).unsqueeze(0)
  token_ids = token_ids.to(device)
  model.eval()
  for _ in range(max_tokens):
    input_ids = token_ids[:,-context_length:]
    with torch.no_grad():
      logits = model(input_ids)
      probs = torch.softmax(logits[:,-1,:],dim=-1)
      next_id = torch.argmax(probs,dim=-1,keepdim=True)
      token_ids = torch.cat([token_ids,next_id],dim=-1)
  text = tokenizer.decode(token_ids.squeeze(0).tolist())
  return text
      
            
def train(epochs,model,train_dataloader,val_dataloader,loss_func,optimizer,eval_prompt,tokenizer):
  
    train_loss = []
    val_loss = []
    tokens_seen = []
    tokens = 0
    for epoch in range(epochs):
      model.train()
      running_train_loss = 0
      running_val_loss = 0
      
      for x,y in tqdm(train_dataloader):
        
        x,y = x.to(device),y.to(device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
          logits = model(x)
          loss = loss_func(logits.flatten(0,1),y.flatten()) # (b,s,d) -> (b*s,d)
          
        running_train_loss += loss.item()
        tokens += x.numel()
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
      train_loss.append(running_train_loss/len(train_dataloader))
      tokens_seen.append(tokens)
      print(f"train_loss after {epoch+1} epochs {running_train_loss/len(train_dataloader):.4f}")
      
      model.eval()
      for x_val,y_val in tqdm(val_dataloader):
        
        x_val,y_val = x_val.to(device),y_val.to(device)
        with torch.no_grad():
          logits = model(x_val)
          
        loss = loss_func(logits.flatten(0,1),y_val.flatten())
        running_val_loss += loss.item()
        
      val_loss.append(running_val_loss/len(val_dataloader))
      print(f"val_loss after {epoch+1} epochs {running_val_loss/len(val_dataloader):.4f}")
      
      #evaluating the perplexity
      perplexity = torch.exp(torch.tensor(running_val_loss / len(val_dataloader)))
      print(f"Perplexity after {epoch+1} epochs: {perplexity.item():.4f}")
      
      print(f'saving the model at {epoch}')
      torch.save(model.state_dict(),'gpt_model.pth')
      
      # evaluation on prompt
      text_output = text_generater(tokenizer,eval_prompt,context_length,max_tokens)
      print("text_generated:")
      print(text_output)
      
    return train_loss,val_loss,tokens_seen
      
train_loss,val_loss,tokens_seen = train(epochs,model,train_dataloader,val_dataloader,loss_func,
                                        optimizer,eval_prompt,tokenizer)
  




