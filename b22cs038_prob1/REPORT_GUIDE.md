# Report Writing Guide (For This Project)

Use this guide to quickly convert your code outputs into a clean assignment report.

## 1. Title and Objective

Suggested title:
- "Learning Domain-Specific Word Embeddings with CBOW and Skip-gram on IIT Jodhpur Corpus"

Suggested objective paragraph:
- This work builds neural word embeddings from institutional text using Word2Vec-style CBOW and Skip-gram with negative sampling. The objective is to study semantic quality under different embedding dimensions, context windows, and numbers of negative samples via nearest-neighbor retrieval, analogy tests, and low-dimensional visualization.

## 2. Dataset Preparation (Task 1)

Mention these points:
1. Source data: raw PDF files in `data/raw/`
2. Cleaning steps used in code:
   - Remove HTML tags, URLs, emails, page numbering, and copyright text
   - Keep alphabetic tokens only
   - Remove stop words
   - Lowercase all tokens
3. Tokenization tool: spaCy (`en_core_web_sm`)
4. Outputs:
   - `data/cleaned_corpus.txt`
   - `outputs/plots/wordcloud.png` (if generated)

Add dataset statistics from preprocessing logs:
- Number of documents
- Total tokens
- Vocabulary size

## 3. Methodology (Task 2)

### 3.1 Model Formulation

Briefly include:
- CBOW predicts center word from surrounding context
- Skip-gram predicts context words from center word
- Both optimized using negative sampling

You can add the objective intuition:
- Positive pair score should be high
- Random negative pair scores should be low

### 3.2 Training Setup

From `config.py`:
- Embedding dimension $N \in \{50, 100, 300\}$
- Context window $C \in \{2, 5, 8\}$
- Negative samples $k \in \{5, 10, 15\}$
- Learning rate: 0.025
- Epochs: 5
- Batch size: 256
- Minimum frequency threshold: 5

Total setting combinations per architecture:
$$
3 \times 3 \times 3 = 27
$$

## 4. Evaluation Protocol (Task 3)

### 4.1 Nearest Neighbor Queries
Use the fixed query words in `evaluation.py`:
- research
- student
- phd
- exam

For each model, report top-5 nearest neighbors using cosine similarity.

### 4.2 Analogy Tests
Use analogies implemented in code:
1. ug : btech :: pg : ?
2. student : hostel :: faculty : ?
3. culture : ignus :: sports : ?

Describe vector arithmetic used:
$$
\vec{d} \approx \vec{b} - \vec{a} + \vec{c}
$$

## 5. Visualization (Task 4)

Include both:
- t-SNE plot
- PCA plot

Current cluster words used by code:
- research, phd, thesis, student, btech, exam, grade, faculty, professor, fest, hostel, campus, library

Discuss whether related terms cluster together and whether any noise/outliers are visible.

## 6. Results Tables You Can Copy

### Table A: Hyperparameter Summary

| Model | Dim | Window | Neg Samples | Checkpoint Present | Evaluated |
|------|-----|--------|-------------|--------------------|-----------|
| CBOW | 50/100/300 | 2/5/8 | 5/10/15 | Yes/No | Yes/No |
| Skip-gram | 50/100/300 | 2/5/8 | 5/10/15 | Yes/No | Yes/No |

### Table B: Qualitative Neighbor Quality (Example Format)

| Model | Dim | Win | Neg | Query | Top Neighbors (Top-3 shown) | Quality Comment |
|------|-----|-----|-----|-------|-------------------------------|-----------------|
| CBOW | 100 | 5 | 10 | student | scholar, students, received | Mostly relevant |

### Table C: Analogy Quality (Example Format)

| Model | Dim | Win | Neg | Analogy | Top Prediction | Acceptable? | Notes |
|------|-----|-----|-----|---------|----------------|-------------|-------|
| CBOW | 100 | 5 | 10 | student:hostel::faculty:? | ... | Yes/No | ... |

## 7. Discussion Prompts

Use these prompts for analysis:
1. Does increasing embedding dimension improve semantic coherence consistently?
2. How does larger context window affect topical vs syntactic similarity?
3. What changes when negative samples increase from 5 to 15?
4. Are CBOW and Skip-gram behaviorally different on this corpus?
5. Why might some outputs look noisy (domain mismatch, corpus size, OCR artifacts, token filtering)?

## 8. Limitations to Mention

1. Some query words may be missing from vocabulary after min-frequency filtering (for example, `exam` can be OOV in some runs).
2. Training uses only corpus-internal unsupervised signals; no external benchmarks are used.
3. Quality judgments are mostly qualitative (neighbors, analogies, plots).

## 9. Reproducibility Checklist

Before finalizing report, verify:
1. `config.py` values match the settings reported.
2. `outputs/evaluation_report.txt` corresponds to the checkpoints discussed.
3. Figures inserted in report match files under `outputs/plots/`.
4. Mention if Skip-gram was trained in current run or loaded from pre-existing checkpoints.

## 10. Suggested Final Report Structure

1. Introduction
2. Dataset and Preprocessing
3. Models and Training Strategy
4. Experimental Setup
5. Results (Neighbors + Analogies + Plots)
6. Comparative Analysis (CBOW vs Skip-gram)
7. Limitations and Future Work
8. Conclusion
