import torch
import torch.nn as nn
from typing import Tuple

class VanillaRNN(nn.Module):
    """
    This model processes sequences character by character, updating a hidden state
    that acts as the 'memory' of the network.
    """
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        """
        Initializes the Vanilla RNN model.

        Args:
            vocab_size (int): The size of the vocabulary (number of unique characters).
            embed_size (int): The size of the embedding vector for each character.
            hidden_size (int): The size of the hidden state vector.
        """
        super(VanillaRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Embedding layer to convert character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # RNN Core Weights
        # W_ih: Weights transforming the input (x_t) to the hidden state space
        self.W_ih = nn.Linear(embed_size, hidden_size)

        # W_hh: Weights transforming the previous hidden state (h_{t-1}) to the new hidden state space
        self.W_hh = nn.Linear(hidden_size, hidden_size)

        # Output Weights to transform the hidden state to the output space (vocab size)
        self.W_ho = nn.Linear(hidden_size, vocab_size)

        # Activation function
        self.tanh = nn.Tanh()
    
    def forward(self, input_seq: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RNN.
        Args:
            input_seq: Input sequence tensor of shape (batch_size, sequence_length)
            hidden_state: Initial hidden state of shape (batch_size, hidden_size)
        Returns:
            outputs: Predictions for each time step of shape (batch_size, seq_len, vocab_size)
            hidden_state: The final hidden state
        """

        batch_size, seq_len = input_seq.size()

        # Convert character indices to embeddings
        embedded = self.embedding(input_seq)  # Shape: (batch_size, seq_len, embed_size)

        # Pre-compute the input-to-hidden transformation
        x_projected = self.W_ih(embedded)  # Shape: (batch_size, seq_len, hidden_size)

        # Create a tensor to store the output predictions at each time step
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(input_seq.device)

        for t in range(seq_len):
            x_t_proj = x_projected[:, t, :]  # Shape: (batch_size, embed_size)

            # THE CORE RNN EQUATION: h_t = tanh(W_ih * x_t + W_hh * h_{t-1})
            hidden_state = self.tanh(x_t_proj + self.W_hh(hidden_state))  # Update hidden state

            out_t = self.W_ho(hidden_state) # Compute output for the current time step
            outputs[:, t, :] = out_t  # Store the output
        
        return outputs, hidden_state
    
    def init_hidden(self, batch_size: int, device: str) -> torch.Tensor:
        """
        Initializes the hidden state with zeros at the start of a new sequence.
        """
        return torch.zeros(batch_size, self.hidden_size).to(device)

    def count_params(self) -> int:
        """
        Counts the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)