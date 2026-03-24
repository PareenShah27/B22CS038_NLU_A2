import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import VanillaRNN, BiLSTM, AttentionRNN
from typing import Dict, Tuple, List, TypeAlias

ModelInstance: TypeAlias = VanillaRNN | BiLSTM | AttentionRNN
ModelClass: TypeAlias = type[VanillaRNN] | type[BiLSTM] | type[AttentionRNN]

# ==========================================
# 1. HELPER: REBUILD VOCABULARY
# ==========================================
def load_vocab(file_path: str) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """
    Rebuilds the exact character-to-index mapping used during training.
    """
    with open(file_path, 'r', encoding = 'utf-8') as f:
        names = [line.strip().lower() for line in f.readlines()]
    
    chars = set(''.join(names))
    charMap = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}

    for i, char in enumerate(sorted(list(chars))):
        charMap[char] = i + 3
    
    idxMap = {i : char for char, i in charMap.items()}

    return charMap, idxMap, names

# ==========================================
# 2. AUTOREGRESSIVE GENERATION
# ==========================================
def generate_name(model: ModelInstance, charMap: Dict[str, int], idxMap: Dict[int, str], device: torch.device, max_len: int = 50, temperature: float = 1.0, min_len: int = 2, train_names: List[str] = None) -> str:
    """
    Generates a single name character-by-character.
    The temperature parameter controls the randomness of predictions.
    Higher temperature = more diverse (but potentially less realistic) names.
    """
    model.eval()

    # Start the sequence with the <SOS> token
    curr_seq = [charMap['<SOS>']]

    # For recurrent decoders, maintain state across generation steps.
    if isinstance(model, VanillaRNN):
        hidden = model.init_hidden(1, device)
    elif isinstance(model, BiLSTM):
        # Only the forward LSTM state is needed at generation time.
        # fc_out is trained on h_fwd only, so generation is fully causal.
        h_fwd = torch.zeros(1, model.hidden_size, device=device)
        c_fwd = torch.zeros(1, model.hidden_size, device=device)
    
    with torch.no_grad():
        for _ in range(max_len):
            # Prepare the input tensor depending on model type.

            if isinstance(model, VanillaRNN):
                x = torch.tensor([[curr_seq[-1]]], dtype=torch.long).to(device)
                outputs, hidden = model(x, hidden)
                next_char_logits = outputs[0,0,:]
            elif isinstance(model, BiLSTM):
                # Step one token through the forward LSTM cell.
                # fc_out is trained on h_fwd only, so no backward state needed.
                x = torch.tensor([curr_seq[-1]], dtype=torch.long).to(device)
                x_t = model.embedding(x)
                h_fwd, c_fwd = model.lstm_cell(
                    x_t, h_fwd, c_fwd, model.W_ih_fwd, model.W_hh_fwd
                )
                next_char_logits = model.fc_out(h_fwd).squeeze(0)
            else:
                # AttentionRNN can decode from the current prefix directly.
                x = torch.tensor([curr_seq], dtype=torch.long).to(device)
                outputs = model(x)
                next_char_logits = outputs[0, -1, :]

            # Apply temperature scaling to control diversity/realism
            next_char_logits = next_char_logits.clone() / temperature

            # Do not sample control tokens as generated characters.
            next_char_logits[charMap['<PAD>']] = float('-inf')
            next_char_logits[charMap['<SOS>']] = float('-inf')

            # Avoid immediate termination with very short outputs.
            if len(curr_seq) - 1 < min_len:
                next_char_logits[charMap['<EOS>']] = float('-inf')

            # Convert logits to probabilities
            probs = F.softmax(next_char_logits, dim=0).cpu().numpy()

            # Sample the next character from the probability distribution
            next_char_idx = np.random.choice(len(probs), p=probs)
            
            # Stop if the model predicts the End of Sequence token
            if next_char_idx == charMap['<EOS>']:
                break

            curr_seq.append(next_char_idx)
            
    # Convert indices back to string, ignoring the <SOS> token
    generated_name = "".join([idxMap[idx] for idx in curr_seq[1:]])
    return generated_name

# ==========================================
# 3. EVALUATION SCRIPT
# ==========================================

def evaluate(model_path: str, model_class: ModelClass, vocab_size: int, embed_size: int, hidden_size: int, charMap: Dict[str, int], idxMap: Dict[int, str], train_names: List[str], device: torch.device, num_samples: int = 1000, temperature: float = 1.0) -> Tuple[float, float]:
    """
    Loads the trained model and generates a specified number of names.
    Evaluates the generated names against the training set to check for duplicates.
    """

    # 1. Initialize and load the model weights (.pth)
    model = model_class(vocab_size, embed_size, hidden_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"\nEvaluating {model_class.__name__}...")

    # 2. Generate names
    generated_names = []
    for _ in range(num_samples):
        # We use a slight temperature tweak to balance realism and diversity
        name = generate_name(model, charMap, idxMap, device, temperature=0.8, train_names=train_names)
        # Avoid empty strings if the model fails and immediately outputs <EOS>
        if len(name.strip()) > 1:
            generated_names.append(name)
    
    # Save all generated names
    with open(f"{model_class.__name__}_generated.txt", "w", encoding="utf-8") as f:
        for name in generated_names:
            f.write(name + "\n")
    
    # 3. Quantitative Evaluation (Novelty & Diversity)
    train_set = set(train_names)
    total_valid_gen = len(generated_names)
    
    unique_generated = set(generated_names)
    duplicate_names = [name for name in unique_generated if name in train_set]
    novel_names = [name for name in unique_generated if name not in train_set]

    # Save novel names
    with open(f"{model_class.__name__}_novel.txt", "w", encoding="utf-8") as f:
        for name in novel_names:
            f.write(name + "\n")

    # Save duplicates (names already in training set)
    with open(f"{model_class.__name__}_duplicates.txt", "w", encoding="utf-8") as f:
        for name in duplicate_names:
            f.write(name + "\n")
    
    with open(f"{model_class.__name__}_unique.txt", "w", encoding="utf-8") as f:
        for name in unique_generated:
            f.write(name + "\n")
    
    diversity = len(unique_generated) / total_valid_gen if total_valid_gen > 0 else 0
    novelty_rate = (len(novel_names) / total_valid_gen) * 100 if total_valid_gen > 0 else 0

    print(f"--- Quantitative Evaluation ---")
    print(f"Diversity Score: {diversity:.4f}")
    print(f"Novelty Rate:    {novelty_rate:.2f}%")
    
    # 4. Task 3: Qualitative Analysis (Realism & Samples)
    print(f"--- Qualitative Samples ---")
    print(f"Randomly selected generated names for your report:")
    sample_display = list(unique_generated)[:10]
    for s in sample_display:
        print(f" - {s.title()}")
        
    return diversity, novelty_rate

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Must match your training hyperparameters
    EMBED_SIZE = 32
    HIDDEN_SIZE = 128
    EVAL = {
        'VanillaRNN': True,
        'BiLSTM': True,
        'AttentionRNN': True
    }
    
    charMap, idxMap, train_names = load_vocab("Indian_names.txt")
    vocab_size = len(charMap)

    try:
        if EVAL['VanillaRNN']:
            evaluate('vanilla_rnn.pth', VanillaRNN, vocab_size, EMBED_SIZE, HIDDEN_SIZE, charMap, idxMap, train_names, device)
        
        if EVAL['BiLSTM']:
            evaluate('bilstm.pth', BiLSTM, vocab_size, EMBED_SIZE, HIDDEN_SIZE, charMap, idxMap, train_names, device)
            
        if EVAL['AttentionRNN']:
            evaluate('attention_rnn.pth', AttentionRNN, vocab_size, EMBED_SIZE, HIDDEN_SIZE, charMap, idxMap, train_names, device)
                       
    except FileNotFoundError as e:
        print("\n[!] Error: Could not find the .pth files. Make sure you run train.py and save the models first!")