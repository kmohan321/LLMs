import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
import tiktoken as tk
import wandb
from tqdm.auto import tqdm

from gpt_model import GPT
from .train_config import config
from data import train_loader, val_loader

wandb.init(project="gpt2-custom-training", config=config)
device = config['device']
tokenizer = tk.get_encoding('gpt2')

model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scaler = GradScaler(device=device)
loss_func = nn.CrossEntropyLoss()

# Log model gradients and topology
wandb.watch(model, log="all", log_freq=100)

def generate_text(model, tokenizer, prompt, max_tokens, context_len):
    model.eval()
    token_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    for _ in range(max_tokens):
        input_ids = token_ids[:, -context_len:]
        with torch.no_grad():
            logits = model(input_ids)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            token_ids = torch.cat([token_ids, next_id], dim=-1)
    return tokenizer.decode(token_ids.squeeze(0).tolist())

print(f"Starting training on {device}...")

for epoch in range(config['epochs']):
    model.train()
    running_train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_func(logits.flatten(0, 1), y.flatten())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_train_loss += loss.item()
        
        wandb.log({"step_train_loss": loss.item()})
        pbar.set_postfix({'loss': loss.item()})

    avg_train_loss = running_train_loss / len(train_loader)
    
    model.eval()
    running_val_loss = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits = model(x_val)
            v_loss = loss_func(logits.flatten(0, 1), y_val.flatten())
            running_val_loss += v_loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_val_loss))
    generated_sample = generate_text(
        model, tokenizer, config['eval_prompt'], config['max_gen_tokens'], config['context_length']
    )
    
    wandb.log({
        "epoch": epoch + 1,
        "avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
        "perplexity": perplexity.item(),
        "generated_text": wandb.Html(f"<p><b>Prompt:</b> {config['eval_prompt']}<br/><b>Result:</b> {generated_sample}</p>")
    })
    
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), 'gpt_model.pth')

wandb.finish()