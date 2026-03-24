import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_Attention(nn.Module):
    """
    This model uses an additive (Bahdanau-style) self-attention mechanism where
    the prediction of the next character depends on a dynamically weighted sum 
    of all previously generated hidden states.
    """
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        """
        Initializes RNN with Bahdanau-style Attention Model

        Args:
            vocab_size (int): The size of the vocabulary (number of unique characters).
            embed_size (int): The size of the embedding vector for each character.
            hidden_size (int): The size of the hidden state vector for each direction.
        """

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Standard RNN Core Weights
        self.W_ih = nn.Linear(embed_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)

        # Weights for Attention Mechanism
        self.W_attn_curr = nn.Linear(hidden_size, hidden_size)  # For current hidden state
        self.W_attn_past = nn.Linear(hidden_size, hidden_size)  # For past hidden state
        self.v_attn = nn.Linear(hidden_size, 1, bias=False)  # To compute attention scores 

        # Final Output Projection
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size) # Concatenated hidden state and context vector
    
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass for RNN with Self-Attention
        
        Args:
            input_seq: Input sequence tensor of shape (batch_size, sequence_length)
        Returns:
            outputs: Predictions for each time step of shape (batch_size, seq_len, vocab_size)
        """

        batch_size, seq_len = input_seq.size()
        embedded = self.embedding(input_seq)
        
        # Tensor to store final outputs across the sequence
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(input_seq.device)
        
        # Initial hidden state
        h_t = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)

        # List to keep track of past hidden states for the attention mechanism
        past_hidden_states = []

        for t in range(seq_len):
            x_t = embedded[:, t, :]
            
            # --- Update the Current Hidden State ---
            h_t = torch.tanh(self.W_ih(x_t) + self.W_hh(h_t))

            # --- Compute Attention Context ---
            if t == 0:
                # At the first time step, there is no past to attend to.
                context = torch.zeros_like(h_t)
            else:
                # Stack all past hidden states: Shape -> (batch_size, t, hidden_size)
                H_past = torch.stack(past_hidden_states, dim=1)
                
                # Expand h_t to match dimensions for broadcasting: Shape -> (batch_size, 1, hidden_size)
                h_t_expanded = h_t.unsqueeze(1)
                
                # Calculate alignment scores (Energy)
                energy = torch.tanh(self.W_attn_curr(h_t_expanded) + self.W_attn_past(H_past))
                
                # Project down to a single score per past time step: Shape -> (batch_size, t)
                scores = self.v_attn(energy).squeeze(2)
                
                # Apply Softmax to get probability weights (Alpha)
                alphas = F.softmax(scores, dim=1) # Shape -> (batch_size, 1, t)
                
                # Compute the context vector as a weighted sum of past hidden states
                context = torch.bmm(alphas.unsqueeze(1), H_past).squeeze(1) # Shape -> (batch_size, hidden_size)
            
            # Save the current hidden state for future time steps to look back at
            past_hidden_states.append(h_t)

            # --- Combine and Predict ---
            # Concatenate the context vector (what we attended to) with h_t (where we are now)
            combined = torch.cat((context, h_t), dim=1)
            
            # Predict the next character distribution
            outputs[:, t, :] = self.fc_out(combined)
        
        return outputs

    def count_parameters(self) -> int:
        """
        Counts the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
