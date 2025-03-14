#
# Copyright © 2025 by Neparth
#
#---------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from collections import Counter
import random
import math

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.word_counts = Counter()
        self.vocab_built = False
    
    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        for text in texts:
            self.word_counts.update(text.split())
        
        # Get most common words
        most_common = self.word_counts.most_common(self.vocab_size - 4)  # -4 for special tokens
        for word, _ in most_common:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab_built = True
        print(f"Vocabulary built with {len(self.word_to_idx)} tokens")
    
    def encode(self, text, add_special_tokens=True):
        """Convert text to token IDs"""
        if not self.vocab_built:
            raise ValueError("Vocabulary has not been built yet")
        
        tokens = []
        if add_special_tokens:
            tokens.append(self.word_to_idx["<BOS>"])
        
        for word in text.split():
            tokens.append(self.word_to_idx.get(word, self.word_to_idx["<UNK>"]))
        
        if add_special_tokens:
            tokens.append(self.word_to_idx["<EOS>"])
        
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Convert token IDs back to text"""
        words = []
        for idx in token_ids:
            word = self.idx_to_word.get(idx, "<UNK>")
            if skip_special_tokens and word in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
                continue
            words.append(word)
        
        return " ".join(words)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.max_length:
                # Create multiple examples by sliding window
                for i in range(0, len(tokens) - self.max_length + 1, self.max_length // 2):
                    self.examples.append(tokens[i:i+self.max_length])
            else:
                # Pad to max_length
                self.examples.append(tokens + [0] * (self.max_length - len(tokens)))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Create input and target by offsetting by 1
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=4, 
                 num_heads=8, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(embed_size, vocab_size)
        
        self.embed_size = embed_size
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None):
        embedded = self.embedding(src) * math.sqrt(self.embed_size)
        embedded = self.pos_encoder(embedded)
        transformer_output = self.transformer_encoder(embedded, src_mask)
        output = self.output_layer(transformer_output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def train_model(model, train_dataloader, val_dataloader, epochs=10, lr=0.001, device='cuda'):
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Reshape for cross entropy loss
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": train_loss / (batch_idx + 1)})
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Reshape for cross entropy loss
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_dataloader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pt')
            print(f"Model saved with validation loss: {val_loss:.4f}")
    
    return model


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, device='cuda'):
    model.eval()
    
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_tensor)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Apply softmax to convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, 1).item()
            
            # If we hit the end of sequence token, stop
            if next_token == tokenizer.word_to_idx["<EOS>"]:
                break
            
            # Add the next token to the input
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(input_tensor[0].cpu().tolist())
    return generated_text


def load_dataset(data_path, test_split=0.1):
    """Load text data from a file and split into train and validation sets"""
    with open(data_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    # Clean the texts
    texts = [text.strip() for text in texts if text.strip()]
    
    # Split into train and validation
    random.shuffle(texts)
    split_idx = int(len(texts) * (1 - test_split))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    return train_texts, val_texts


def main():
    # Parameters
    data_path = "/home/neparth/neparth_workers/servers_daemons/sites-101s_servers/dataset.txt"  # Path to your text file
    vocab_size = 10000
    embed_size = 734
    hidden_size = 512
    num_layers = 4
    num_heads = 8
    batch_size = 32
    epochs = 10
    lr = 0.001
    max_length = 128
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare data
    train_texts, val_texts = load_dataset(data_path)
    
    # Initialize tokenizer and build vocabulary
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(train_texts)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length=max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=max_length)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = TransformerModel(
        vocab_size=len(tokenizer.word_to_idx),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # Train the model
    model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        lr=lr,
        device=device
    )
    
    # Generate sample text
    sample_prompts = [
        "0 + 0 =",
    ]
    
    for prompt in sample_prompts:
        generated_text = generate_text(model, tokenizer, prompt, temperature=0.8, device=device)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()

#---------------------------------------------------------------------------------------------------------------------------------