"""
Handles Task-3 and Task-4

Responsibilities:
1. Loads trained model embeddings.
2. Computes top-5 nearest neighbors using cosine similarity.
3. Performs semantic analogy experiments (e.g., A : B :: C : D).
4. Visualizes word clusters using PCA or t-SNE.

"""

import json
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import config
from models import CBOW, SkipGram

# ==========================================
# 1. UTILITY: LOADING EMBEDDINGS
# ==========================================

def load_vocab(load_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Loads the vocabulary mappings from JSON files.
    JSON converts integer keys to strings, so we must cast idx2word keys back to integers.
    """
    # Load word2idx
    with open(os.path.join(load_dir, 'word2idx.json'), 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    
    # Load idx2word and convert keys to int
    with open(os.path.join(load_dir, 'idx2word.json'), 'r', encoding='utf-8') as f:
        raw_idx2word = json.load(f)
        idx2word = {int(k): v for k, v in raw_idx2word.items()}
    
    print(f"[IO] Vocabulary loaded from {load_dir}")
    return word2idx, idx2word



def extract_embeddings(model_path: str, model_type: str, vocab_size: int, embed_dim: int) -> np.ndarray:
    """
    Loads the saved PyTorch model and extracts the embedding matrix as a NumPy array.
    """
    device = torch.device("cpu")

    if model_type == 'cbow':
        model = CBOW(vocab_size, embed_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        target_emb = model.target_embeddings.weight.data
        context_emb = model.context_embeddings.weight.data
        embeddings = (target_emb + context_emb).numpy()
    elif model_type == 'skipgram':
        model = SkipGram(vocab_size, embed_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        target_emb = model.target_embeddings.weight.data
        context_emb = model.context_embeddings.weight.data
        embeddings = (target_emb + context_emb).numpy()
    else:
        raise ValueError("Invalid model_type. Use 'cbow' or 'skipgram'.")

    return embeddings

def get_random_word_embedding(
    embeddings: np.ndarray,
    idx2word: Dict[int, str],
    exclude_pad: bool = True,
    seed: int | None = None
) -> Tuple[str, np.ndarray]:
    """
    Returns a random in-vocabulary word and its embedding vector.

    Args:
        embeddings (np.ndarray): Embedding matrix of shape (vocab_size, embed_dim).
        idx2word (dict): Mapping from index to token.
        exclude_pad (bool): Excludes the '<PAD>' token when True.
        seed (int | None): Optional seed for reproducible sampling.
    """
    rng = random.Random(seed)

    valid_indices = [idx for idx in idx2word.keys() if 0 <= idx < len(embeddings)]
    if exclude_pad:
        valid_indices = [idx for idx in valid_indices if idx2word[idx] != '<PAD>']

    if not valid_indices:
        raise ValueError("No valid vocabulary indices available for random sampling.")

    chosen_idx = rng.choice(valid_indices)
    chosen_word = idx2word[chosen_idx]
    chosen_vector = embeddings[chosen_idx]

    return chosen_word, chosen_vector

def format_embedding_line(word: str, vector: np.ndarray, precision: int = 4) -> str:
    """
    Formats a word embedding as: word - v1, v2, v3, ...
    """
    values = ", ".join(f"{float(v):.{precision}f}" for v in vector)
    return f"{word} - {values}"

def get_word_embedding_by_word(
    word: str,
    embeddings: np.ndarray,
    word2idx: Dict[str, int]
) -> np.ndarray:
    """
    Returns embedding vector for a specific word.
    """
    lookup = word.lower().strip()
    if lookup not in word2idx:
        raise KeyError(f"Word '{word}' not found in vocabulary.")
    return embeddings[word2idx[lookup]]

# ==========================================
# 2. TASK-3: SEMANTIC ANALYSIS
# ==========================================

def get_top_n_neighbors(target_word: str, embeddings: np.ndarray, word2idx: Dict[str, int], idx2word: Dict[int, str], n: int = 5) -> List:
    """
    Finds the top N nearest neighbors for a given word using Cosine Similarity.
    
    Args:
        target_word (str): The word to query.
        embeddings (np.ndarray): The trained embedding matrix.
        word2idx (dict): Vocabulary mapping.
        idx2word (dict): Inverse vocabulary mapping.
        n (int): Number of neighbors to return.
    """

    if target_word not in word2idx:
        print(f"Word '{target_word}' not in vocabulary.")
        return []
    
    target_idx = word2idx[target_word]
    target_vec = embeddings[target_idx].reshape(1, -1)

    # Compute cosine similarity between the target vector and ALL vectors in the vocabulary
    similarities = cosine_similarity(target_vec, embeddings)[0]

    # Sort indices by descending similarity, excluding the word itself (which will be at index 0)
    sorted_indices = np.argsort(similarities)[::-1][1:n+1]
    results = [(idx2word[idx], similarities[idx]) for idx in sorted_indices]

    print(f"\nTop {n} nearest neighbors for '{target_word}':")
    for word, sim in results:
        print(f" - {word} (Similarity: {sim:.4f})")
    
    return results

def eval_analogy(A: str, B: str, C: str, embeddings: np.ndarray, word2idx: Dict[str, int], idx2word: Dict[int, str], n: int = 3) -> List:
    """
    Solves analogies of the form: A is to B as C is to ?.
    Vector math: Vector(?) = Vector(B) - Vector(A) + Vector(C)
    """

    for w in [A, B, C]:
        if w not in word2idx:
            print(f"Analogy failed: '{w}' not in vocabulary.")
            return []
    
    # Extract Vectors
    vecA = embeddings[word2idx[A]]
    vecB = embeddings[word2idx[B]]
    vecC = embeddings[word2idx[C]]

    # Compute the expected vector and simillarities
    expected_vec = (vecB - vecA + vecC).reshape(1, -1)
    similarities = cosine_similarity(expected_vec, embeddings)[0]

    # Filter out input words
    input_indices = {word2idx[A], word2idx[B], word2idx[C]}
    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in sorted_indices:
        if idx not in input_indices and idx != 0:
            results.append((idx2word[idx], similarities[idx]))
            if len(results) == n:
                break
    
    print(f"Analogy: {A} : {B} :: {C} : ?")
    for word, sim in results:
        print(f" -> {word} (Similarity: {sim:.4f})")
    
    return results

# ==========================================
# 3. TASK-4: VISUALIZATION
# ==========================================

def plot_embeddings(embeddings: np.ndarray, word2idx: Dict[str, int], idx2word: Dict[int, str], word_to_plot: List[str] | None = None, num_words: int = 300, method: str = 'tsne', save_name: str = 'plot.png') -> None:
    """
    Projects high-dimensional embeddings down to 2D using PCA or t-SNE and plots them.
    """
    print(f"\nProjecting embeddings to 2D using {method.upper()}...")

    # If no specific words are provided, we just take the top 'num_words' most frequent words
    if word_to_plot is None:
        indices = list(range(1, min(num_words + 1, len(word2idx))))
        labels = [idx2word[idx] for idx in indices]
    else:
        indices = [word2idx[w] for w in word_to_plot if w in word2idx]
        labels = [w for w in word_to_plot if w in word2idx]
    vectors = embeddings[indices]

    # Apply Dimensionality Reduction
    if method == 'tsne':
        n_samples = len(indices)
        reducer = TSNE(n_components=2, random_state=config.SEED, perplexity=n_samples // 3)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=config.SEED)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    reduced_vectors = reducer.fit_transform(vectors)

    # Plotting
    plt.figure(figsize=(14, 10))
    for i, label in enumerate(labels):
        x, y = reduced_vectors[i,:]
        plt.scatter(x, y, marker='x', color='red')
        plt.annotate(label, xy = (x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=9)
    
    plt.title(f"Word Embeddings 2D Projection ({method.upper()})")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(config.PLOTS_DIR, save_name)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved visualization to: {save_path}")
    plt.close()

# ==========================================
# 4. EVALUATION
# ==========================================

def evaluate(model_path: str, model_type: str, vocab_size: int, embed_dim: int, word2idx: Dict[str, int], idx2word: Dict[int, str]) -> None:
    print(f"\n{'='*40}\nEvaluating Model: {model_path}\n{'='*40}")

    # 1. Extract Embeddings
    embeddings = extract_embeddings(model_path, model_type, vocab_size, embed_dim)

    # 2. Nearest Neighbors
    target_words = ['research', 'student', 'phd', 'exam']
    for word in target_words:
        get_top_n_neighbors(word, embeddings, word2idx, idx2word)
    
    # 3. Analogy Experiments
    # Example 1
    eval_analogy('ug', 'btech', 'pg', embeddings, word2idx, idx2word)
    # Example 2
    eval_analogy('student', 'hostel', 'faculty', embeddings, word2idx, idx2word)
    # Example 3
    eval_analogy('culture', 'ignus', 'sports', embeddings, word2idx, idx2word)

    # 4. Visualization
    cluster_words = ['research', 'phd', 'thesis', 'student', 'btech', 'exam', 'grade', 'faculty', 'professor', 'fest', 'hostel', 'campus', 'library']
    plot_embeddings(embeddings, word2idx, idx2word, cluster_words, method='tsne', save_name=f"{model_type}_tsne.png")
    plot_embeddings(embeddings, word2idx, idx2word, cluster_words, method='pca', save_name=f"{model_type}_pca.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate embeddings and print word vectors.")
    parser.add_argument('--model-type', type=str, default='skipgram', choices=['cbow', 'skipgram'])
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--neg-samples', type=int, default=10)
    parser.add_argument('--word', type=str, default=None, help='Specific word to print embedding for.')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Seed used for random-word sampling.')
    parser.add_argument('--vector-only', action='store_true', help='Print vector only, skip full evaluation.')
    args = parser.parse_args()

    # 1. Load the saved vocabulary
    word2idx, idx2word = load_vocab(config.MODELS_DIR)
    vocab_size = len(word2idx)

    # 2. Define the exact hyperparameter combo we want to evaluate
    embed_dim = args.embed_dim
    model_type = args.model_type
    model_filename = f"{model_type}_dim{embed_dim}_win{args.window_size}_neg{args.neg_samples}.pt"
    model_path = os.path.join(config.MODELS_DIR, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # 3. Run the evaluation tasks
    if not args.vector_only:
        evaluate(
            model_path=model_path,
            model_type=model_type,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            word2idx=word2idx,
            idx2word=idx2word
        )

    # 4. Print embedding in form-ready format.
    embeddings = extract_embeddings(model_path, model_type, vocab_size, embed_dim)
    if args.word:
        try:
            vector = get_word_embedding_by_word(args.word, embeddings, word2idx)
            print("\nSpecific word embedding (copy-paste format):")
            print(format_embedding_line(args.word.lower().strip(), vector, precision=4))
        except KeyError as e:
            print(f"\n{e}")
    else:
        rand_word, rand_vector = get_random_word_embedding(embeddings, idx2word, exclude_pad=True, seed=args.seed)
        print("\nRandom word embedding (copy-paste format):")
        print(format_embedding_line(rand_word, rand_vector, precision=4))