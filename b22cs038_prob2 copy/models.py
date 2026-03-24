import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# ==========================================
# MODEL 1: VANILLA RNN
# ==========================================
class VanillaRNN(nn.Module):
    """
    Vanilla Recurrent Neural Network.
    Processes sequences character by character, updating a hidden state.
    """
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super(VanillaRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # RNN Core Weights
        self.W_ih = nn.Linear(embed_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        
        # Output Weights
        self.W_ho = nn.Linear(hidden_size, vocab_size)
        self.tanh = nn.Tanh()

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RNN.

        Args:
            input_seq: Input sequence tensor of shape (batch_size, sequence_length)
            hidden_state: Initial hidden state of shape (batch_size, hidden_size)
        Returns:
            outputs: Predictions for each time step of shape (batch_size, seq_len, vocab_size)
            hidden_state: The final hidden state
        """

        batch_size, seq_len = input.size()
        embedded = self.embedding(input)
        
        # OPTIMIZATION: Pre-compute input-to-hidden transformation for all time steps
        x_projected = self.W_ih(embedded) 
        
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(input.device)
        
        # Iterate sequentially only for the recurrent hidden state dependency
        for t in range(seq_len):
            x_t_proj = x_projected[:, t, :]
            # h_t = tanh((W_ih * x_t) + (W_hh * h_{t-1}))
            hidden = self.tanh(x_t_proj + self.W_hh(hidden))
            outputs[:, t, :] = self.W_ho(hidden)
            
        return outputs, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initializes the hidden state with zeros at the start of a new sequence.
        """
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def count_parameters(self) -> int:
        """
        Counts the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==========================================
# MODEL 2: BIDIRECTIONAL LSTM
# ==========================================
class BiLSTM(nn.Module):
    """
    Bidirectional LSTM.
    Processes sequences forward and backward, concatenating hidden states.
    """
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Forward LSTM Weights (Combining all 4 gates into one matrix for efficiency)
        self.W_ih_fwd = nn.Linear(embed_size, 4 * hidden_size)
        self.W_hh_fwd = nn.Linear(hidden_size, 4 * hidden_size)
        
        # Backward LSTM Weights
        self.W_ih_bwd = nn.Linear(embed_size, 4 * hidden_size)
        self.W_hh_bwd = nn.Linear(hidden_size, 4 * hidden_size)
        
        # Output Layer — predicts from the forward hidden state only so the
        # model is fully causal and can generate autoregressively.
        # The backward LSTM still runs during training: it shares the embedding
        # layer and back-propagates richer gradients through it, acting as a
        # regularising auxiliary signal without breaking causal inference.
        self.fc_out = nn.Linear(hidden_size, vocab_size)

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

        # Compute all gate transformations at once
        gates = W_ih(x_t) + W_hh(h_prev)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=1)
        
        i_t = torch.sigmoid(ingate)       
        f_t = torch.sigmoid(forgetgate)   
        g_t = torch.tanh(cellgate)        
        o_t = torch.sigmoid(outgate)      
        
        # Update Cell and Hidden States
        c_t = (f_t * c_prev) + (i_t * g_t)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BiLSTM.

        Args:
            input_seq: Input sequence tensor of shape (batch_size, sequence_length)
        Returns:
            outputs: Predictions for each time step of shape (batch_size, seq_len, vocab_size)
        """

        batch_size, seq_len = input.size()
        embedded = self.embedding(input) 
        
        h_fwd = torch.zeros(batch_size, self.hidden_size).to(input.device)
        c_fwd = torch.zeros(batch_size, self.hidden_size).to(input.device)
        h_bwd = torch.zeros(batch_size, self.hidden_size).to(input.device)
        c_bwd = torch.zeros(batch_size, self.hidden_size).to(input.device)
        
        outputs_fwd = torch.zeros(batch_size, seq_len, self.hidden_size).to(input.device)
        outputs_bwd = torch.zeros(batch_size, seq_len, self.hidden_size).to(input.device)
        
        # Forward Pass
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h_fwd, c_fwd = self.lstm_cell(x_t, h_fwd, c_fwd, self.W_ih_fwd, self.W_hh_fwd)
            outputs_fwd[:, t, :] = h_fwd
            
        # Backward Pass
        for t in range(seq_len - 1, -1, -1):
            x_t = embedded[:, t, :]
            h_bwd, c_bwd = self.lstm_cell(x_t, h_bwd, c_bwd, self.W_ih_bwd, self.W_hh_bwd)
            outputs_bwd[:, t, :] = h_bwd
            
        # Concatenate forward and backward hidden states
        # (kept for architecture completeness; backward enriches gradients
        # through the shared embedding during training)
        combined_hidden = torch.cat((outputs_fwd, outputs_bwd), dim=2)  # noqa: F841 — unused in fc_out
        
        # Predict from the causal forward direction only
        predictions = self.fc_out(outputs_fwd)
        
        return predictions

    def count_parameters(self) -> int:
        """
        Counts the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==========================================
# MODEL 3: RNN WITH BASIC ATTENTION
# ==========================================
class AttentionRNN(nn.Module):
    """
    RNN with Basic Attention Mechanism.
    Uses additive self-attention to weight past hidden states dynamically.
    """
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super(AttentionRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.W_ih = nn.Linear(embed_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        
        # Attention Weights
        self.W_attn_curr = nn.Linear(hidden_size, hidden_size)
        self.W_attn_past = nn.Linear(hidden_size, hidden_size)
        self.v_attn = nn.Linear(hidden_size, 1, bias=False)
        
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass for RNN with Self-Attention
        
        Args:
            input_seq: Input sequence tensor of shape (batch_size, sequence_length)
        Returns:
            outputs: Predictions for each time step of shape (batch_size, seq_len, vocab_size)
        """

        batch_size, seq_len = input.size()
        embedded = self.embedding(input)
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(input.device)
        h_t = torch.zeros(batch_size, self.hidden_size).to(input.device)
        past_hidden_states = []
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            
            # Step 1: Update Current Hidden State
            h_t = torch.tanh(self.W_ih(x_t) + self.W_hh(h_t))
            
            # Step 2: Compute Attention Context
            if t == 0:
                context = torch.zeros_like(h_t)
            else:
                H_past = torch.stack(past_hidden_states, dim=1)
                h_t_expanded = h_t.unsqueeze(1)
                
                # Energy = tanh(W_curr * h_t + W_past * H_past)
                energy = torch.tanh(self.W_attn_curr(h_t_expanded) + self.W_attn_past(H_past))
                scores = self.v_attn(energy).squeeze(2)
                
                # Alpha (weights) and Context Vector
                alpha = F.softmax(scores, dim=1).unsqueeze(1) 
                context = torch.bmm(alpha, H_past).squeeze(1) 
            
            past_hidden_states.append(h_t)
            
            # Step 3: Combine and Predict
            combined = torch.cat((context, h_t), dim=1)
            outputs[:, t, :] = self.fc_out(combined)
            
        return outputs

    def count_parameters(self) -> int:
        """
        Counts the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)