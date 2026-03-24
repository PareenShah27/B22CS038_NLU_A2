import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models import VanillaRNN, BiLSTM, AttentionRNN
from typing import Tuple, List
import numpy as np

# ==========================================
# 1. DATASET PREPARATION
# ==========================================
class NameDataset(Dataset):
    """
    Custom PyTorch Dataset to load Indian names, build the character vocabulary,
    and convert strings to integer tensors.
    """
    def __init__(self, file_path: str, max_len: int = 50):
        # Read the generated names
        with open(file_path, 'r', encoding = 'utf-8') as f:
            self.names = [line.strip().lower() for line in f.readlines()]
        
        self.max_len = max_len

        # Build Vocabulary: Unique characters + Special Tokens
        chars = set(''.join(self.names))
        self.charMap = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}  # Special tokens

        # Assign an integer to each unique character
        for i, ch in enumerate(sorted(list(chars))):
            self.charMap[ch] = i + 3
        
        self.idxMap = {idx: char for char, idx in self.charMap.items()}
        self.vocab_size = len(self.charMap)

    def __len__(self) -> int:
        return len(self.names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx]

        # Convert characters to indices and add <SOS> and <EOS>
        indices = [self.charMap['<SOS>']] + [self.charMap[c] for c in name] + [self.charMap['<EOS>']]
        
        # Truncate if too long, pad if too short
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
            indices[-1] = self.charMap['<EOS>'] # Ensure it ends with EOS
        else:
            indices += [self.charMap['<PAD>']] * (self.max_len - len(indices))
        
        # The input is the sequence except the last character
        # The target is the sequence except the first character (shifted by 1)
        x = torch.tensor(indices[:-1], dtype=torch.long)
        y = torch.tensor(indices[1:], dtype=torch.long)

        return x, y

# ==========================================
# 2. THE TRAINING LOOP
# ==========================================
def train_model(model: VanillaRNN | BiLSTM | AttentionRNN, dataloader: DataLoader, epochs: int, learning_rate: float, clip_value: float = 5.0, device: torch.device = torch.device('cpu')):
    """
    Executes the training loop for a given sequence model.
    """
    model.to(device)

    # We use CrossEntropyLoss but ignore the <PAD> token so it doesn't skew the loss
    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {model.__class__.__name__}...")
    print(f"Trainable Parameters: {model.count_parameters()}")

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            batch_size = x_batch.size(0)

            optimizer.zero_grad()
            
            # Forward pass: Check if the model requires a hidden state initialization
            if isinstance(model, VanillaRNN):
                hidden = model.init_hidden(batch_size, device)
                outputs, _ = model(x_batch, hidden)
            else:
                # BiLSTM and Attention handle their own hidden state initialization internally
                outputs = model(x_batch)
            
            # Flatten outputs and targets for CrossEntropyLoss
            outputs = outputs.view(-1, model.vocab_size)
            y_batch = y_batch.view(-1)

            loss = criterion(outputs, y_batch)

            # Backward pass (Backpropagation Through Time)
            loss.backward()
            
            # Gradient Clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Task 1 Requirement: Specify Hyperparameters
    BATCH_SIZE = 64
    MAX_LEN = 20
    EMBED_SIZE = 32
    HIDDEN_SIZE = 128
    EPOCHS = 30
    LEARNING_RATE = 0.005
    TRAIN = {
        'VanillaRNN': True,
        'BiLSTM': True,
        'AttentionRNN': True
    }
    
    # Load Data
    dataset = NameDataset("Indian_names.txt", max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    vocab_size = dataset.vocab_size
    
    print(f"Vocabulary Size: {vocab_size}")
    
    # --- Instantiate Models ---
    # Task 1 requires comparing these three models
    rnn_model = VanillaRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
    blstm_model = BiLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
    attention_model = AttentionRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
    
    # --- Train Models ---
    # Save the trained model state dictionaries for the generation phase
    if TRAIN['VanillaRNN']:
        train_model(rnn_model, dataloader, EPOCHS, LEARNING_RATE, device=device)
        torch.save(rnn_model.state_dict(), "vanilla_rnn.pth")
    if TRAIN['BiLSTM']:
        train_model(blstm_model, dataloader, EPOCHS, LEARNING_RATE, device=device)
        torch.save(blstm_model.state_dict(), "bilstm.pth")
    if TRAIN['AttentionRNN']:
        train_model(attention_model, dataloader, EPOCHS, LEARNING_RATE, device=device)
        torch.save(attention_model.state_dict(), "attention_rnn.pth")

            