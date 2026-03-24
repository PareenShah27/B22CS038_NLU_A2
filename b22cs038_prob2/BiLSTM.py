import torch
import torch.nn as nn
from typing import Tuple

class BiLSTM(nn.Module):
    """
    It processes sequences in both forward and backward directions and concatenates
    their hidden states to predict the next character.
    """
    
    def _init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        """
        Initializes the BiLSTM model.

        Args:
            vocab_size (int): The size of the vocabulary (number of unique characters).
            embed_size (int): The size of the embedding vector for each character.
            hidden_size (int): The size of the hidden state vector for each direction.
        """
        super(BiLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Embedding layer to convert character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # --- FORWARD LSTM WEIGHTS ---
        # Instead of 4 separate linear layers, we optimize by doing one large projection of 
        # size 4 * hidden_size, which we will later split into the 4 gates (i, f, g, o).
        self.W_ih_fwd = nn.Linear(embed_size, 4 * hidden_size)
        self.W_hh_fwd = nn.Linear(hidden_size, 4 * hidden_size)

        # --- BACKWARD LSTM WEIGHTS ---
        self.W_ih_bwd = nn.Linear(embed_size, 4 * hidden_size)
        self.W_hh_bwd = nn.Linear(hidden_size, 4 * hidden_size)

        # Output Weights to transform the concatenated hidden states to the output space (vocab size)
        self.fc_out = nn.Linear(2 * hidden_size, vocab_size)
    
    def lstm_cell(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor, W_ih: nn.Linear, W_hh: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the LSTM cell operations for a single time step.

        Args:
            x_t: Input at time t (batch_size, embed_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            c_prev: Previous cell state (batch_size, hidden_size)
            W_ih: Linear layer for input-to-hidden transformation
            W_hh: Linear layer for hidden-to-hidden transformation
        Returns:
            h_t: New hidden state (batch_size, hidden_size)
            c_t: New cell state (batch_size, hidden_size)
        """

        # Compute the gates and candidate cell state
        gates = W_ih(x_t) + W_hh(h_prev)  # Shape: (batch_size, 4 * hidden_size)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, dim=1)  # Split into 4 parts

        # Apply activations
        i_t = torch.sigmoid(in_gate)        # Input gate : What new info to store
        f_t = torch.sigmoid(forget_gate)    # Forget gate : What past info to discard
        g_t = torch.tanh(cell_gate)         # Candidate cell : New candidate values
        o_t = torch.sigmoid(out_gate)       # Output gate : What parts of cell state to output

        # Update cell state and hidden state
        c_t = (f_t * c_prev) + (i_t * g_t)  # New cell state
        h_t = o_t * torch.tanh(c_t)         # New hidden state

        return h_t, c_t
    
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BiLSTM.
        Args:
            input_seq: Input sequence tensor of shape (batch_size, sequence_length)
        Returns:
            outputs: Predictions for each time step of shape (batch_size, seq_len, vocab_size)
        """

        batch_size, seq_len = input_seq.size()

        # Convert character indices to embeddings
        embedded = self.embedding(input_seq)  # Shape: (batch_size, seq_len, embed_size)

        # Initialize hidden and cell states for both directions
        h_fwd = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)
        c_fwd = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)

        h_bwd = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)
        c_bwd = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)

        # Tensors to store hidden states for both directions
        h_fwd_seq = torch.zeros(batch_size, seq_len, self.hidden_size).to(input_seq.device)
        h_bwd_seq = torch.zeros(batch_size, seq_len, self.hidden_size).to(input_seq.device)

        # --- FORWARD PASS ---
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # Shape: (batch_size, embed_size)
            h_fwd, c_fwd = self.lstm_cell(x_t, h_fwd, c_fwd, self.W_ih_fwd, self.W_hh_fwd)
            h_fwd_seq[:, t, :] = h_fwd  # Store forward hidden state

        # --- BACKWARD PASS ---
        for t in range(seq_len - 1, -1, -1):
            x_t = embedded[:, t, :]  # Shape: (batch_size, embed_size)
            h_bwd, c_bwd = self.lstm_cell(x_t, h_bwd, c_bwd, self.W_ih_bwd, self.W_hh_bwd)
            h_bwd_seq[:, t, :] = h_bwd  # Store backward hidden state
        
        # Concatenate forward and backward hidden states
        combined = torch.cat((h_fwd_seq, h_bwd_seq), dim=2)  # Shape: (batch_size, seq_len, 2 * hidden_size)

        # Pass the combined hidden states through the final linear layer
        predictions = self.fc_out(combined)  # Shape: (batch_size, seq_len, vocab_size)

        return predictions

    def count_parameters(self) -> int:
        """
        Counts the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)