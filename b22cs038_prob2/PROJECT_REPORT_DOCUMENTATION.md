# Character-Level Indian Name Generation: Project Documentation

## 1. Project Scope and Submission Note

This project builds a character-level name generator for Indian names using PyTorch.

Three sequence models are implemented and compared:
- VanillaRNN
- BiLSTM (causal decoding from forward state)
- AttentionRNN (additive self-attention over past hidden states)

Important submission note:
- The final implementation used for submission is consolidated in `models.py`.
- Files `RNN.py`, `BiLSTM.py`, and `RNN_Attention.py` represent earlier standalone building blocks and are not the final integrated model definition file.

## 2. Problem Statement

Given a corpus of Indian names, train autoregressive character-level language models that can generate realistic and diverse new names.

Primary goals:
- Learn name-like character sequences.
- Generate novel names not copied directly from the training set.
- Compare architectures on diversity and novelty.

## 3. Repository Structure

Core execution files:
- `models.py`: final integrated model definitions (VanillaRNN, BiLSTM, AttentionRNN).
- `train.py`: dataset pipeline and training loop.
- `gen_eval.py`: generation and evaluation (quantitative + qualitative outputs).
- `names_cleaner.py`: utility to deduplicate `Indian_names.txt`.

Data and artifacts:
- `Indian_names.txt`: training data (1000 lines in current workspace).
- `vanilla_rnn.pth`, `bilstm.pth`, `attention_rnn.pth`: trained checkpoints.
- `*_generated.txt`, `*_unique.txt`, `*_novel.txt`, `*_duplicates.txt`: evaluation outputs per model.

## 4. Data Pipeline and Preprocessing

Implemented in `train.py` (`NameDataset`):

1. Read and lowercase each name.
2. Build vocabulary from all characters in the corpus.
3. Add special tokens:
   - `<PAD>` = 0
   - `<SOS>` = 1
   - `<EOS>` = 2
4. Convert each name to index sequence:
   - `[<SOS>] + characters + [<EOS>]`
5. Truncate/pad to a fixed maximum length (`MAX_LEN = 20`).
6. Create input-target pairs by 1-step shift:
   - Input `x`: sequence except last token
   - Target `y`: sequence except first token

Why this works:
- The 1-step shift transforms name generation into next-character prediction.
- Ignoring `<PAD>` in loss avoids training signal from artificial padding positions.

## 5. Model Architectures (Final Unified File: `models.py`)

### 5.1 VanillaRNN

Components:
- Embedding layer
- Manual recurrent transition:
  - hidden update with `tanh(W_ih x_t + W_hh h_{t-1})`
- Output projection `W_ho` to vocabulary logits

Notes:
- Uses explicit hidden-state initialization (`init_hidden`).
- Forward loop is time-step sequential, but input projection is precomputed for efficiency.

### 5.2 BiLSTM

Components:
- Shared embedding
- Manual LSTM cell implementation for both forward and backward passes
- Gate computations in combined matrices (`4 * hidden_size` per direction)

Causal decoding decision:
- Although both directions are computed in training, output logits are produced from forward hidden states only (`fc_out(outputs_fwd)`).
- This keeps generation autoregressive and causally valid.
- Backward pass still provides gradient signal through shared embeddings during training.

### 5.3 AttentionRNN

Components:
- RNN hidden update (similar to VanillaRNN)
- Additive attention over all past hidden states
  - current state interacts with each past state via learned projections
  - softmax scores yield attention weights
  - weighted context vector is concatenated with current hidden state
- Final output layer consumes `[context ; h_t]`

Effect:
- Allows dynamic emphasis on different previous positions while predicting the next character.

## 6. Training Methodology

Implemented in `train.py` (`train_model`):

Hyperparameters used:
- Batch size: 64
- Max length: 20
- Embedding size: 32
- Hidden size: 128
- Epochs: 30
- Learning rate: 0.005
- Optimizer: Adam
- Loss: CrossEntropyLoss with `ignore_index=0` (`<PAD>`)
- Gradient clipping: max norm = 5.0

Device:
- Automatically uses CUDA if available, else CPU.

Training procedure:
1. Load batches from DataLoader.
2. Compute logits (model-specific forward path).
3. Flatten logits and targets for token-level cross-entropy.
4. Backpropagate through time.
5. Apply gradient clipping.
6. Update parameters.
7. Save checkpoint per model.

## 7. Generation and Evaluation Protocol

Implemented in `gen_eval.py`:

Generation:
- Starts from `<SOS>`.
- Samples one character at a time from softmax distribution.
- Temperature scaling is applied (configured as 0.8 during evaluation calls).
- `<PAD>` and `<SOS>` are blocked during sampling.
- `<EOS>` is blocked for very short prefixes (`min_len=2`) to avoid empty outputs.

Evaluation outputs per model:
- `*_generated.txt`: all valid generated names.
- `*_unique.txt`: unique generated names.
- `*_novel.txt`: unique names not in training set.
- `*_duplicates.txt`: unique names that overlap with training set.

Metrics:
- Diversity score = `|unique_generated| / |generated|`
- Novelty rate (%) = `100 * |novel| / |generated|`

## 8. Quantitative Results (Current Workspace Artifacts)

The following values were computed from the generated artifact files in this workspace:

| Model | Generated | Unique | Novel | Duplicates | Diversity | Novelty Rate |
|---|---:|---:|---:|---:|---:|---:|
| VanillaRNN | 1000 | 984 | 948 | 36 | 0.984 | 94.80% |
| BiLSTM | 1000 | 951 | 870 | 81 | 0.951 | 87.00% |
| AttentionRNN | 1000 | 936 | 872 | 64 | 0.936 | 87.20% |

Dataset size observed:
- `Indian_names.txt`: 1000 lines

## 9. Qualitative Observations (Sample Outputs)

### 9.1 Novel Samples (examples)

VanillaRNN samples:
- viney kriyanshu
- tumbarshi
- manankar singh
- jendala garg

BiLSTM samples:
- deepansh manoj
- rajpureer singh
- dharam sharma
- sidharth babijar

AttentionRNN samples:
- neelkar pranavi
- shagu kamal raj
- saumitra songiyar
- kritin raj

### 9.2 Duplicate Samples (examples seen in training set)

VanillaRNN duplicates:
- yogesh kumar
- priyanshu
- amit kumar

BiLSTM duplicates:
- avani rai
- deepak singh
- kanika singh

AttentionRNN duplicates:
- ram prasad
- aman sharma
- chanchal

## 10. Interpretation of Results

1. VanillaRNN achieved the highest measured diversity and novelty in this run.
2. BiLSTM and AttentionRNN produced lower duplicate rates than might be expected for more expressive models only in some settings; here they still show strong novelty but slightly reduced diversity versus VanillaRNN.
3. Richer architectures can overfit frequent local patterns in relatively small datasets if regularization/tuning is not extensive.

Important nuance:
- These metrics are run-dependent because sampling is stochastic and no fixed random seed is enforced in generation.

## 11. Limitations

1. Single train/eval split with no held-out validation metrics (e.g., perplexity).
2. No multi-seed averaging; results may vary across runs.
3. Character-level modeling can generate orthographically valid but semantically odd names.
4. Output quality is sensitive to temperature and max-length constraints.

## 12. Suggested Improvements

1. Add deterministic seeds for reproducible comparisons.
2. Track validation loss/perplexity and apply early stopping.
3. Perform temperature sweep (e.g., 0.6, 0.8, 1.0) with metric curves.
4. Add additional quality checks:
   - invalid character rate
   - average name length distribution
   - edit-distance to nearest training name
5. Evaluate multiple runs and report mean plus standard deviation.

## 13. Reproducibility Guide

### 13.1 Environment

Install dependencies:
- Python 3.10+ (recommended)
- `torch`
- `numpy`

### 13.2 Run Order

1. Optional cleaning:
   - Run `names_cleaner.py` to deduplicate the source list.
2. Train models:
   - Run `train.py`.
3. Generate and evaluate:
   - Run `gen_eval.py`.
4. Collect report artifacts:
   - Use all `*_generated.txt`, `*_unique.txt`, `*_novel.txt`, `*_duplicates.txt` files.

## 14. Report-Ready Writeup Template

Use the following structure directly in your assignment report:

1. Introduction
   - Task: Indian name generation via character-level neural language models.
   - Motivation: model sequence structure and creativity vs memorization.
2. Data and Preprocessing
   - Describe vocabulary creation, special tokens, fixed length, and shifted targets.
3. Model Designs
   - Explain VanillaRNN, BiLSTM (causal output), AttentionRNN.
4. Training Setup
   - Hyperparameters, optimizer, loss, gradient clipping, device.
5. Evaluation Method
   - Define diversity and novelty mathematically.
   - Explain generated/unique/novel/duplicate files.
6. Results
   - Insert quantitative table from Section 8.
   - Add selected qualitative examples from Section 9.
7. Discussion
   - Analyze why models differ and role of data size/sampling.
8. Limitations and Future Work
   - Include reproducibility and robustness improvements.
9. Conclusion
   - Summarize key empirical findings and practical takeaways.

## 15. Academic Integrity and Claim Boundaries

When presenting results:
- State clearly that metrics are computed from current saved artifacts.
- Avoid claiming universal superiority from a single run.
- Mention stochasticity and dataset-size effects.

---

This document is designed to be directly reusable for a detailed assignment report with minimal editing.
