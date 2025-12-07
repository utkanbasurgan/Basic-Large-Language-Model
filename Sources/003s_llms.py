#
# Utkan Başurgan
#
#---------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TextDataset(Dataset):
    
    def __init__(self, text, seq_length):
        
        self.words = text.split()
        self.vocab = sorted(list(set(self.words)))
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        self.seq_length = seq_length
        
    def __len__(self):
        
        return len(self.words) - self.seq_length
    
    def __getitem__(self, idx):
        
        input_seq = [self.word_to_idx[w] for w in self.words[idx:idx + self.seq_length]]
        target_seq = [self.word_to_idx[w] for w in self.words[idx + 1:idx + self.seq_length + 1]]
        return torch.tensor(input_seq), torch.tensor(target_seq)


class AttentionLayer(nn.Module):
    
    def __init__(self, hidden_size):
        
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, gru_output):
        
        attention_weights = torch.softmax(self.attention(gru_output), dim=1)
        context = torch.sum(attention_weights * gru_output, dim=1)
        return context, attention_weights


class GRUWithAttention(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.3):
        
        super(GRUWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.gru = nn.GRU(
            embed_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = AttentionLayer(hidden_size * 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
    def forward(self, x):
        
        embedded = self.embedding(x)
        embedded = self.dropout1(embedded)
        
        gru_out, _ = self.gru(embedded)
        gru_out = self.layer_norm(gru_out)
        
        context, attention_weights = self.attention(gru_out)
        context = self.dropout2(context)
        
        out = self.fc1(context)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out.unsqueeze(1).repeat(1, x.size(1), 1), attention_weights


def train_model(model, dataloader, epochs, device, save_interval=5):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2
    )
    
    best_loss = float('inf')
    
    model.train()
    for epoch in range(epochs):
        
        total_loss = 0
        batch_losses = []
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_losses.append(loss.item())
            
            if batch_idx % 25 == 0 and batch_idx > 0:
                
                avg_batch_loss = np.mean(batch_losses[-25:])
                print(f"  Batch {batch_idx}, Loss: {avg_batch_loss:.4f}")
                
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_loss < best_loss:
            
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_gru_model.pth')
            print(f"  Saved best model (loss: {best_loss:.4f})")
            
        if (epoch + 1) % save_interval == 0:
            
            torch.save(model.state_dict(), f'gru_model_epoch_{epoch+1}.pth')
            
        print("-" * 60)


def generate_text(model, dataset, start_text, length, device, top_k=5):
    
    model.eval()
    words = start_text.split()
    
    with torch.no_grad():
        
        for _ in range(length):
            
            input_words = words[-dataset.seq_length:] if len(words) >= dataset.seq_length else words
            input_indices = [dataset.word_to_idx.get(w, 0) for w in input_words]
            
            while len(input_indices) < dataset.seq_length:
                
                input_indices.insert(0, 0)
                
            x = torch.tensor([input_indices]).to(device)
            output, _ = model(x)
            
            logits = output[0, -1]
            top_probs, top_indices = torch.topk(torch.softmax(logits, dim=0), top_k)
            
            selected_idx = top_indices[torch.multinomial(top_probs, 1)].item()
            next_word = dataset.idx_to_word[selected_idx]
            
            words.append(next_word)
            
    return ' '.join(words)


def main():
    
    with open('input.txt', 'r', encoding='utf-8') as f:
        
        text = f.read()
    
    seq_length = 32
    batch_size = 48
    embed_size = 200
    hidden_size = 300
    num_layers = 3
    dropout = 0.3
    epochs = 20
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    dataset = TextDataset(text, seq_length)
    print(f"Vocabulary size: {len(dataset.vocab)}")
    print(f"Total words: {len(dataset.words)}")
    print(f"Training samples: {len(dataset)}\n")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = GRUWithAttention(
        len(dataset.vocab), 
        embed_size, 
        hidden_size, 
        num_layers,
        dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")
    print("="*60)
    
    train_model(model, dataloader, epochs, device)
    
    print("\n" + "="*60)
    print("GENERATION EXAMPLES")
    print("="*60)
    
    for i, seed in enumerate(["The", "In the", "Once upon"], 1):
        
        print(f"\nExample {i} (seed: '{seed}', top_k=5):")
        generated = generate_text(model, dataset, seed, 50, device, top_k=5)
        print(generated)
        
    for i, seed in enumerate(["The", "In the"], 1):
        
        print(f"\nExample {i+3} (seed: '{seed}', top_k=10):")
        generated = generate_text(model, dataset, seed, 50, device, top_k=10)
        print(generated)
        
    torch.save(model.state_dict(), 'final_gru_model.pth')


if __name__ == '__main__':
    
    main()

#---------------------------------------------------------------------------------------------------------------------------------
