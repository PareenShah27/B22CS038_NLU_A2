"""
Contains the neural network architectures.

Includes:
1. CBOW: Continuous Bag of Words architecture predicting a center word from context using Negative Sampling.
2. SkipGram: Skip-gram architecture optimized using Negative Sampling.

"""

import torch
import torch.nn as nn
from typing import Tuple

class CBOW(nn.Module):
    """
    Continuous Bag of Words (CBOW) Model with Negative Sampling.
    Predicts a target (center) word based on the surrounding context words.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super(CBOW, self).__init__()

        # Define the embedding layer for target words
        self.target_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        # Define the embedding layer for context words (used for negative sampling)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        # Initialize weights uniformly
        init_range = 0.5 / embed_dim
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, target_indices: torch.Tensor, context_indices: torch.Tensor, negative_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing the raw dot products for positive and negative pairs.
        We output raw dot products to be used with PyTorch's BCEWithLogitsLoss.
        Args:
            target_idx (Tensor): Indices of the target words. Shape: (batch_size, 1)
            context_idx (Tensor): Indices of the true context words. Shape: (batch_size, num_context_words)
            negative_indices (Tensor): Indices of the negative samples. Shape: (batch_size, num_negative_samples)
        Returns:
            positive_score (Tensor): Dot product of target and true context. Shape: (batch_size, 1)
            negative_score (Tensor): Dot products of target and negative samples. Shape: (batch_size, num_negative_samples)
        """

        # Get embeddings for the current batch
        if target_indices.dim() == 1:
            target_indices = target_indices.unsqueeze(1)
        
        context_embeds = self.context_embeddings(context_indices)
        context_vec = torch.mean(context_embeds, dim=1).unsqueeze(1)

        # Get target and negative embeddings
        target_embeds = self.target_embeddings(target_indices)
        neg_embeds = self.target_embeddings(negative_indices)

        # Calculate Scores
        positive_score = torch.bmm(context_vec, target_embeds.transpose(1, 2)).squeeze(2)
        negative_score = torch.bmm(context_vec, neg_embeds.transpose(1, 2)).squeeze(1)

        return positive_score, negative_score
    
    def get_word_embeddings(self) -> torch.Tensor:
        return self.target_embeddings.weight.data + self.context_embeddings.weight.data

class SkipGram(nn.Module):
    """
    Skip-gram Model with Negative Sampling.
    Instead of predicting context words over the entire vocabulary, this model uses a binary 
    classification approach: maximizing the dot product of true (target, context) pairs 
    and minimizing it for generated (target, negative_sample) pairs.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super(SkipGram, self).__init__()

        # Define the embedding layer for target words
        self.target_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        # Define the embedding layer for context words (used for negative sampling)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        # Initialize weights uniformly
        init_range = 0.5 / embed_dim
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, target_indices: torch.Tensor, context_indices: torch.Tensor, negative_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing the raw dot products for positive and negative pairs.
        We output raw dot products to be used with PyTorch's BCEWithLogitsLoss.
        
        Args:
            target_idx (Tensor): Indices of the target words. Shape: (batch_size, 1)
            context_idx (Tensor): Indices of the true context words. Shape: (batch_size, 1)
            negative_indices (Tensor): Indices of the negative samples. Shape: (batch_size, num_negative_samples)
            
        Returns:
            positive_score (Tensor): Dot product of target and true context. Shape: (batch_size, 1)
            negative_score (Tensor): Dot products of target and negative samples. Shape: (batch_size, num_negative_samples)
        """
        # 1. Get embeddings for the current batch
        if target_indices.dim() == 1:
            target_indices = target_indices.unsqueeze(1)
        if context_indices.dim() == 1:
            context_indices = context_indices.unsqueeze(1)

        target_emb = self.target_embeddings(target_indices)
        context_emb = self.context_embeddings(context_indices)
        neg_emb = self.context_embeddings(negative_indices)

        # 2. Calculate Positive Scores
        positive_score = torch.bmm(target_emb, context_emb.transpose(1, 2)).squeeze(2)

        # 3. Calculate Negative Scores
        negative_score = torch.bmm(target_emb, neg_emb.transpose(1, 2)).squeeze(1)

        return positive_score, negative_score
    
    def get_word_embeddings(self) -> torch.Tensor:
        return self.target_embeddings.weight.data + self.context_embeddings.weight.data