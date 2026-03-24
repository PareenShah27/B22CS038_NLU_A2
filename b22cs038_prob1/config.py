"""
Configuration file for Assignment 2 - Problem 1.
Stores all hyperparameters, file paths, and random seeds for reproducibility.
"""

import os

# --- FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_CLEANED_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_corpus.txt')
MODELS_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')

# --- HYPERPARAMETERS FOR EXPERIMENTATION ---

# 1. Embeddings Dimensions (N)
EMBEDDING_DIMS = [50, 100, 300]

# 2. Number of Negative Samples (k)
NEG_SAMPLES = [5, 10, 15] 

# 3. Context Window Size (C)
WINDOW_SIZES = [2, 5, 8]

# --- GENERAL TRAINING PARAMETERS ---
LEARNING_RATE = 0.025
EPOCHS = 5
BATCH_SIZE = 256
MIN_WORD_FREQUENCY = 5 # Standard threshold to drop extremely rare words 
SEED = 27 # For consistent PCA/t-SNE and negative sampling