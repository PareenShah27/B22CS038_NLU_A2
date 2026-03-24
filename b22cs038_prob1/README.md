# CSL 7640 Assignment 2 - Problem 1

This project builds and evaluates Word2Vec-style embeddings on an IIT Jodhpur corpus using:
- CBOW with Negative Sampling
- Skip-gram with Negative Sampling

The codebase is split into four stages:
1. Data preparation from raw PDF files
2. Training over multiple hyperparameter combinations
3. Semantic evaluation (nearest neighbors and analogies)
4. 2D visualization (t-SNE and PCA)

## 1. Project Structure

- `preprocessor.py`: Cleans PDF text, tokenizes with spaCy, writes cleaned corpus, and creates word cloud
- `train.py`: Builds vocabulary, prepares datasets, trains CBOW/Skip-gram, saves model checkpoints
- `evaluation.py`: Loads embeddings, computes nearest neighbors + analogies, generates t-SNE/PCA plots
- `main.py`: Stage controller for full pipeline execution
- `config.py`: Centralized paths and hyperparameters
- `outputs/models/`: Trained `.pt` checkpoints and vocabulary JSON files
- `outputs/plots/`: Visualizations
- `outputs/evaluation_report.txt`: Consolidated evaluation logs

## 2. Environment and Dependencies

Recommended Python version: 3.10+

Install packages:

```bash
pip install torch numpy matplotlib scikit-learn spacy wordcloud PyPDF2
python -m spacy download en_core_web_sm
```

## 3. Data Flow

1. Place raw PDF files inside `data/raw/`.
2. Run preprocessing:
   - Extract text from PDFs
   - Clean text (remove URLs, emails, HTML tags, punctuation, stop words)
   - Tokenize + lowercase
   - Save cleaned lines to `data/cleaned_corpus.txt`
3. Train models from cleaned corpus.
4. Evaluate trained models and generate report/plots.

## 4. Configuration

All key settings live in `config.py`.

Default hyperparameters:
- Embedding dimensions: `[50, 100, 300]`
- Window sizes: `[2, 5, 8]`
- Negative samples: `[5, 10, 15]`
- Learning rate: `0.025`
- Epochs: `5`
- Batch size: `256`
- Min word frequency: `5`
- Seed: `27`

Total combinations per architecture: `3 x 3 x 3 = 27`

## 5. Running the Pipeline

Edit stage flags in `main.py`:
- `RUN_DATA_PREP`
- `RUN_TRAINING`
- `RUN_EVALUATION`

Then run:

```bash
python main.py
```

Or run modules separately:

```bash
python preprocessor.py
python train.py
python evaluation.py
```

## 6. Output Artifacts

After successful execution:
- `data/cleaned_corpus.txt`
- `outputs/models/word2idx.json`, `outputs/models/idx2word.json`
- Model checkpoints named like:
  - `cbow_dim100_win5_neg10.pt`
  - `skipgram_dim300_win8_neg15.pt`
- `outputs/plots/*.png`
- `outputs/evaluation_report.txt`

## 7. Important Notes

1. In `main.py`, by default only evaluation is enabled (`RUN_EVALUATION = True`), while preprocessing/training are disabled.
2. In `train.py`, the Skip-gram training loop is currently commented, so running training as-is will generate CBOW checkpoints only.
3. Existing `outputs/models/` already contains both CBOW and Skip-gram checkpoint files.
4. Evaluation queries are fixed in `evaluation.py`:
   - Neighbors for `research`, `student`, `phd`, `exam`
   - Analogies: `ug:btech::pg:?`, `student:hostel::faculty:?`, `culture:ignus::sports:?`

## 8. Suggested Report Usage

Use the generated artifacts for report sections:
- Dataset prep: stats + word cloud
- Experimental setup: hyperparameter grid and training settings
- Results: nearest-neighbor and analogy outputs from `outputs/evaluation_report.txt`
- Visualization: t-SNE/PCA plots in `outputs/plots/`
- Discussion: effect of dimension/window/negative samples and qualitative semantic quality

A ready-to-use reporting template is provided in `REPORT_GUIDE.md`.
