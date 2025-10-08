#
# Utkan Başurgan
#
#---------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math


class TextDataset(Dataset):
    
    def __init__(self, text, seq_length):
        
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.seq_length = seq_length
        self.text = text
        
    def __len__(self):
        
        return len(self.text) - self.seq_length
    
    def __getitem__(self, idx):
        
        chunk = self.text[idx:idx + self.seq_length + 1]
        input_seq = [self.char_to_idx[ch] for ch in chunk[:-1]]
        target_seq = [self.char_to_idx[ch] for ch in chunk[1:]]
        return torch.tensor(input_seq), torch.tensor(target_seq)


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=5000):
        
        super(PositionalEncoding, self).__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        
        return x + self.pe[:x.size(0)]


class TransformerLLM(nn.Module):
    
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        
        super(TransformerLLM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.d_model = d_model
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        src = self.dropout(src)
        
        if src_mask is None:
            
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
            
        output = self.transformer(src, src_mask)
        output = self.fc_out(output)
        
        return output


def train_model(model, dataloader, epochs, device):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    
    model.train()
    for epoch in range(epochs):
        
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
                
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")


def generate_text(model, dataset, start_text, length, device, temperature=1.0):
    
    model.eval()
    chars = [dataset.char_to_idx[ch] for ch in start_text]
    
    with torch.no_grad():
        
        for _ in range(length):
            
            x = torch.tensor([chars]).to(device)
            output = model(x)
            logits = output[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            chars.append(next_char_idx)
            
    return ''.join([dataset.idx_to_char[idx] for idx in chars])


def main():
    
    with open('input.txt', 'r', encoding='utf-8') as f:
        
        text = f.read()
    
    seq_length = 64
    batch_size = 32
    d_model = 256
    nhead = 8
    num_layers = 4
    dim_feedforward = 1024
    dropout = 0.2
    epochs = 15
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = TextDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TransformerLLM(
        len(dataset.chars), 
        d_model, 
        nhead, 
        num_layers, 
        dim_feedforward,
        dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    train_model(model, dataloader, epochs, device)
    
    print("\n" + "="*50)
    print("Generated text (temperature=0.8):")
    print("="*50)
    generated = generate_text(model, dataset, "The ", 300, device, temperature=0.8)
    print(generated)
    
    print("\n" + "="*50)
    print("Generated text (temperature=1.2):")
    print("="*50)
    generated = generate_text(model, dataset, "The ", 300, device, temperature=1.2)
    print(generated)
    
    torch.save(model.state_dict(), 'transformer_llm_model.pth')


if __name__ == '__main__':
    
    main()

#---------------------------------------------------------------------------------------------------------------------------------
