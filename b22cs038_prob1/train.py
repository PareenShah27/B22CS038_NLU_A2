"""
Handles Task-2: Model Training

Responsibilities:
1. Builds the vocabulary and word-to-index mappings.
2. Generates training data (context, target) using a sliding window.
3. Implements PyTorch Dataset classes for CBOW and Skip-gram.
4. Executes the training loops, computing loss and updating weights via backpropagation.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import random
import json
import os

import config
from models import CBOW, SkipGram

# ==========================================
# 1. VOCABULARY & DATA PREPARATION
# ==========================================

def build_vocab(tokenized_corpus: List[str], min_freq: int) -> Tuple[Dict[str, int], Dict[int, str], Counter]:
    """
    Builds word-to-index and index-to-word mappings, filtering rare words.
    
    Args:
        tokenized_corpus (list): Flat list of all tokens in the corpus.
        min_freq (int): Minimum frequency threshold to keep a word.
        
    Returns:
        word2idx (dict): Mapping of word string to integer index.
        idx2word (dict): Mapping of integer index to word string.
        word_counts (Counter): Frequency of each retained word.
    """

    # Count frequencies of all tokens
    raw_counts = Counter(tokenized_corpus)

    # Filter out rare words and create mappings
    filtered_words = {word: count for word, count in raw_counts.items() if count >= min_freq}

    word2idx = {'<PAD>': 0}
    idx2word = {0: '<PAD>'}

    for idx, word in enumerate(filtered_words.keys(), start=1):
        word2idx[word] = idx
        idx2word[idx] = word

    # Re-calculate counts only for words in our final vocabulary
    word_counts = Counter({w: raw_counts[w] for w in filtered_words})

    return word2idx, idx2word, word_counts

def save_vocab(word2idx: Dict[str, int], idx2word: Dict[int, str], save_dir: str) -> None:
    """
    Saves the vocabulary mappings to JSON files.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save word2idx
    with open(os.path.join(save_dir, 'word2idx.json'), 'w', encoding='utf-8') as f:
        json.dump(word2idx, f, indent=4)
        
    # Save idx2word
    with open(os.path.join(save_dir, 'idx2word.json'), 'w', encoding='utf-8') as f:
        json.dump(idx2word, f, indent=4)
        
    print(f"[IO] Vocabulary successfully saved to {save_dir}")

def create_unigram_table(word_cnts: Counter, word2idx: Dict[str, int], table_size: float = 1e7) -> List[int]:
    """
    Creates a pre-computed table for efficient negative sampling.
    Standard Word2Vec implementation raises word frequencies to the 3/4 power 
    to slightly increase the probability of sampling less frequent words.
    """

    table = []

    # Calculate the denominator: sum of (count^(3/4)) for all words
    pow_freq = np.array(list(word_cnts.values())) ** 0.75
    words_pow_sum = sum(pow_freq)

    # Calculate fractional probability for each word and fill the table
    cnt = 0
    for word, freq in word_cnts.items():
        p = (freq ** 0.75) / words_pow_sum
        table_portion = int(p * table_size)
        table.extend([word2idx[word]] * table_portion)
    
    return table

# ==========================================
# 2. PYTORCH DATASETS
# ==========================================

class CBOWData(Dataset):
    """
    PyTorch Dataset for Continuous Bag of Words.
    Yields (context_indices, target_index).
    """
    def __init__(self, tokenized_corpus: List[str], word2idx: Dict[str, int], window_size: int, num_neg_samples: int, unigram_table: List[int]):
        self.data = []
        self.num_neg_samples = num_neg_samples
        self.unigram_table = unigram_table

        # Convert tokens to indices, ignoring OOV (Out of Vocab) words
        indices = [word2idx[token] for token in tokenized_corpus if token in word2idx]

        for i in range(window_size, len(indices) - window_size):
            context = indices[i - window_size : i] + indices[i + 1 : i + window_size + 1]
            target = indices[i]
            self.data.append((torch.tensor(context), torch.tensor(target)))
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, target = self.data[idx]
        
        neg_samples = []
        while len(neg_samples) < self.num_neg_samples:
            neg_sample = random.choice(self.unigram_table)
            if neg_sample != target and neg_sample not in context:
                neg_samples.append(neg_sample)
        
        return torch.tensor(target), torch.tensor(context), torch.tensor(neg_samples)

class SkipGramData(Dataset):
    """
    PyTorch Dataset for Skip-gram with Negative Sampling.
    Yields (target_index, context_index, negative_indices).
    """
    def __init__(self, tokenized_corpus: List[str], word2idx: Dict[str, int], window_size: int, num_neg_samples: int, unigram_table: List[int]):
        self.data = []
        self.num_neg_samples = num_neg_samples
        self.unigram_table = unigram_table

        indices = [word2idx[token] for token in tokenized_corpus if token in word2idx]

        for i in range(window_size, len(indices) - window_size):
            target = indices[i]
            context_indices = indices[i - window_size : i] + indices[i + 1 : i + window_size + 1]

            for context in context_indices:
                self.data.append((target, context))
            
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target, context = self.data[idx]

        # Generate negative samples for this specific target-context pair
        neg_samples = []
        while len(neg_samples) < self.num_neg_samples:
            neg_sample = random.choice(self.unigram_table)
            if neg_sample != target and neg_sample != context:
                neg_samples.append(neg_sample)
        
        return torch.tensor(target), torch.tensor(context), torch.tensor(neg_samples)
    
# ==========================================
# 3. TRAINING LOOPS
# ==========================================

def train_cbow(dataloader: DataLoader, vocab_size: int, embed_dim: int, epochs: int = config.EPOCHS, lr: float = config.LEARNING_RATE) -> CBOW:
    """
    Trains the CBOW model using CrossEntropyLoss.
    
    Args:
        dataloader (DataLoader): DataLoader for CBOWData.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the embeddings.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimization.
        
    Returns:
        model (CBOW): Trained CBOW model instance.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training CBOW on device: {device}")

    model = CBOW(vocab_size, embed_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SparseAdam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for target, context, neg_samples in dataloader:
            target, context, neg_samples = target.to(device), context.to(device), neg_samples.to(device)

            optimizer.zero_grad() # Zero gradients
            pos_score, neg_score = model(target, context, neg_samples) # Forward pass

            # Create labels: 1 for positive pairs and 0 for negative samples
            pos_labels = torch.ones_like(pos_score, dtype=torch.float32).to(device)
            neg_labels = torch.zeros_like(neg_score, dtype=torch.float32).to(device)

            # Compute loss for positive and negative samples
            loss_pos = criterion(pos_score, pos_labels)
            loss_neg = criterion(neg_score, neg_labels)
            loss = loss_pos + loss_neg

            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
    
    return model

def train_skipgram(dataloader: DataLoader, vocab_size: int, embed_dim: int, epochs: int = config.EPOCHS, lr: float = config.LEARNING_RATE) -> SkipGram:
    """
    Trains the Skip-gram model using Binary Cross-Entropy Loss for Negative Sampling.
    
    Args:
        dataloader (DataLoader): DataLoader for SkipGramData.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimensionality of the embeddings.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimization.
        
    Returns:
        model (SkipGram): Trained Skip-gram model instance.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Skip-gram on device: {device}")

    model = SkipGram(vocab_size, embed_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SparseAdam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for target, context, neg_samples in dataloader:
            target, context, neg_samples = target.to(device), context.to(device), neg_samples.to(device)

            optimizer.zero_grad() # Zero gradients
            pos_score, neg_score = model(target, context, neg_samples) # Forward pass

            # Create labels: 1 for positive pairs and 0 for negative samples
            pos_labels = torch.ones_like(pos_score, dtype=torch.float32).to(device)
            neg_labels = torch.zeros_like(neg_score, dtype=torch.float32).to(device)

            # Compute loss for positive and negative samples
            loss_pos = criterion(pos_score, pos_labels)
            loss_neg = criterion(neg_score, neg_labels)
            loss = loss_pos + loss_neg

            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
    
    return model

def train_pipeline():
    """
    Main execution function for Task-2.
    Loads the cleaned corpus, builds the vocabulary, and runs training loops 
    for both CBOW and Skip-gram across the required hyperparameter combinations.
    """
    print("--- Starting Task 2: Model Training Pipeline ---")

    # 1. Load the cleaned corpus
    if not os.path.exists(config.DATA_CLEANED_PATH):
        raise FileNotFoundError(f"Cleaned corpus not found at {config.DATA_CLEANED_PATH}. Run Task 1 first.")
    
    with open(config.DATA_CLEANED_PATH, 'r', encoding='utf-8') as f:
        corpus_text = f.read()
        corpus_tokens = corpus_text.split()
    
    print(f"Loaded corpus with {len(corpus_tokens)} total tokens.")

    # 2. Build Vocabulary and Unigram Table
    word2idx, idx2word, word_cnts = build_vocab(corpus_tokens, config.MIN_WORD_FREQUENCY)
    vocab_size = len(word2idx)
    print(f"Vocabulary built with size: {vocab_size}")

    # Save the vocabulary mappings
    save_vocab(word2idx, idx2word, config.MODELS_DIR)

    unigram_table = create_unigram_table(word_cnts, word2idx)

    # 3. Experimentation Loops
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    for embed_dim in config.EMBEDDING_DIMS:
        for window_size in config.WINDOW_SIZES:
            # ---------------------------------------------------------
            # Train CBOW Model with Negative Sampling
            # ---------------------------------------------------------
            for neg_samples in config.NEG_SAMPLES:
                print(f"\n[Training CBOW] Dim: {embed_dim}, Window: {window_size}, Neg Samples: {neg_samples}")

                cbow_dataset = CBOWData(corpus_tokens, word2idx, window_size, neg_samples, unigram_table)
                cbow_loader = DataLoader(cbow_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

                cbow_model = train_cbow(
                    dataloader=cbow_loader,
                    vocab_size=vocab_size,
                    embed_dim=embed_dim,
                    epochs=config.EPOCHS,
                    lr=config.LEARNING_RATE
                )

                cbow_file = f"cbow_dim{embed_dim}_win{window_size}_neg{neg_samples}.pt"
                torch.save(cbow_model.state_dict(), os.path.join(config.MODELS_DIR, cbow_file))
                print(f"CBOW model saved to: {cbow_file}")

            # ---------------------------------------------------------
            # Train Skip-gram Model with Negative Sampling
            # ---------------------------------------------------------

            # for neg_samples in config.NEG_SAMPLES:
            #     print(f"\n[Training Skip-gram] Dim: {embed_dim}, Window: {window_size}, Neg Samples: {neg_samples}")

            #     sg_dataset = SkipGramData(corpus_tokens, word2idx, window_size, neg_samples, unigram_table)
            #     sg_loader = DataLoader(sg_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

            #     sg_model = train_skipgram(
            #         dataloader=sg_loader,
            #         vocab_size=vocab_size,
            #         embed_dim=embed_dim,
            #         epochs=config.EPOCHS,
            #         lr=config.LEARNING_RATE
            #     )

            #     # Save the trained model weights
            #     sg_file = f"skipgram_dim{embed_dim}_win{window_size}_neg{neg_samples}.pt"
            #     torch.save(sg_model.state_dict(), os.path.join(config.MODELS_DIR, sg_file))
            #     print(f"Skip-gram model saved to: {sg_file}")
    
    print("\n--- Training Pipeline Completed ---")

if __name__ == "__main__":
    train_pipeline()