#
# Utkan Başurgan
#
#---------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

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


class SimpleLLM(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        
        super(SimpleLLM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


def train_model(model, dataloader, epochs, device):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def generate_text(model, dataset, start_text, length, device):
    
    model.eval()
    chars = [dataset.char_to_idx[ch] for ch in start_text]
    hidden = None
    
    with torch.no_grad():
        
        for _ in range(length):
            
            x = torch.tensor([chars]).to(device)
            output, hidden = model(x, hidden)
            probs = torch.softmax(output[0, -1], dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            chars.append(next_char_idx)
            
    return ''.join([dataset.idx_to_char[idx] for idx in chars])


def main():
    
    with open('input.txt', 'r', encoding='utf-8') as f:
        
        text = f.read()
    
    seq_length = 50
    batch_size = 64
    embed_size = 128
    hidden_size = 256
    num_layers = 2
    epochs = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = TextDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleLLM(len(dataset.chars), embed_size, hidden_size, num_layers).to(device)
    
    train_model(model, dataloader, epochs, device)
    
    generated = generate_text(model, dataset, "The ", 200, device)
    print("\nGenerated text:")
    print(generated)
    
    torch.save(model.state_dict(), 'llm_model.pth')


if __name__ == '__main__':
    
    main()

#---------------------------------------------------------------------------------------------------------------------------------
