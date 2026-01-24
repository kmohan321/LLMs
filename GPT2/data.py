import tiktoken as tk
import torch
from torch.utils.data import Dataset, DataLoader
from .train_config import config

#creating dataset class from pytorch dataset class
class DATASET(Dataset):
  def __init__(self,raw_text,context_length,stride):
    '''
    raw_text -> text extracted from text file
    context_length -> length of sequence of text for input to the model
    stride -> floating window for text (same as stride in convolution)
    # if stride < context_length , may cause overfitting 
    '''
    
    #preparing the tokenizer
    tokenizer = tk.get_encoding('gpt2')
    
    # encoding the text -> text ids
    self.encoded_data = tokenizer.encode(raw_text)
    
    self.input_ids  = []
    self.output_ids = []
    
    for i in range(0,len(self.encoded_data)-context_length,stride):
      
      self.input_ids.append(torch.tensor(self.encoded_data[i:i+context_length]))
      self.output_ids.append(torch.tensor(self.encoded_data[i+1:i+context_length+1]))
      
  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self,idx):
    return self.input_ids[idx] , self.output_ids[idx]

with open(config['file_path'], 'r', encoding="utf-8") as f:
    text = f.read()

split_idx = int(config['train_ratio'] * len(text))
train_data = text[:split_idx]
val_data = text[split_idx:]

train_dataset = DATASET(train_data, config['context_length'], config['stride'])
val_dataset = DATASET(val_data, config['context_length'], config['stride'])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
  