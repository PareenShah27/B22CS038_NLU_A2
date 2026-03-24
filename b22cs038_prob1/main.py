"""
Master execution script

This script completely automates the pipeline:
1. Data Preparation: Parses PDFs/TXTs, tokenizes, and saves the corpus.
2. Training: Trains CBOW and Skip-gram models across all config hyperparameters.
3. Evaluation: Evaluates all trained models (Cosine similarity, analogies) and logs 
   the output to a central text report, plus generates PCA/t-SNE visual plots.
"""

import os
import contextlib
import config

from preprocessor import process_data
from train import train_pipeline
from evaluation import load_vocab, evaluate

def evaluate_all():
    """
    Iterates through all trained models based on config hyperparameters,
    runs the semantic evaluation, and saves the console output to a text file.
    """
    print("\n>>> STAGE 3: Initiating Batch Semantic Analysis & Visualization...")

    # 1. Load the vocabulary mappings saved during training
    try:
        word2idx, idx2word = load_vocab(config.MODELS_DIR)
        vocab_size = len(word2idx)
    except FileNotFoundError:
        print("[Error] Vocabulary files not found. Please run STAGE 2 (Training) first.")
        return
    
    # 2. Setup the output report file
    os.makedirs(os.path.dirname(config.MODELS_DIR), exist_ok=True)
    report_path = os.path.join(os.path.dirname(config.MODELS_DIR), 'evaluation_report.txt')

    print(f"Executing evaluations. Results will be saved to: {report_path}")

    # 3. Open the file and redirect all print statements into it
    with open(report_path, 'w', encoding='utf-8') as f:
        with contextlib.redirect_stdout(f):
            print("="*50)
            print("CSL 7640 - PROBLEM 1 EVALUATION REPORT")
            print("="*50)

            # --- EVALUATE ALL CBOW MODELS ---
            for dim in config.EMBEDDING_DIMS:
                for win in config.WINDOW_SIZES:
                    for neg in config.NEG_SAMPLES:
                        cbow_file = f"cbow_dim{dim}_win{win}_neg{neg}.pt"
                        cbow_path = os.path.join(config.MODELS_DIR, cbow_file)

                        if os.path.exists(cbow_path):
                            evaluate(
                                model_path=cbow_path,
                                model_type='cbow',
                                vocab_size=vocab_size,
                                embed_dim=dim,
                                word2idx=word2idx,
                                idx2word=idx2word,
                            )
                        else:
                            print(f"[Missing Model] {cbow_file} not found.")
            
            # --- EVALUATE ALL SKIP-GRAM MODELS ---
            for dim in config.EMBEDDING_DIMS:
                for win in config.WINDOW_SIZES:
                    for neg in config.NEG_SAMPLES:
                        sg_file = f"skipgram_dim{dim}_win{win}_neg{neg}.pt"
                        sg_path = os.path.join(config.MODELS_DIR, sg_file)

                        if os.path.exists(sg_path):
                            evaluate(
                                model_path=sg_path,
                                model_type='skipgram',
                                vocab_size=vocab_size,
                                embed_dim=dim,
                                word2idx=word2idx,
                                idx2word=idx2word
                            )
                        else:
                            print(f"[Missing Model] {sg_file} not found.")
    print(">>> STAGE 3 Complete. Check the outputs folder for your report and plots.\n")

def main():
    print("="*50)
    print("CSL 7640 - Problem 1")
    print("="*50)

    # ---------------------------------------------------------
    # Pipeline Control Flags
    # Leave all as True for a complete, unattended end-to-end run.
    # ---------------------------------------------------------
    RUN_DATA_PREP = False # Set True for Data Preparation (only needed once)
    RUN_TRAINING = False # Set True for Model Training (only needed once, or if you change hyperparameters)
    RUN_EVALUATION = True

    # ==========================================
    # STAGE 1: DATA PREPARATION
    # ==========================================
    if RUN_DATA_PREP:
        print("\n>>> STAGE 1: Initiating Dataset Preparation...")
        process_data()
        print(">>> STAGE 1 Complete.\n")
    
    # ==========================================
    # STAGE 2: MODEL TRAINING
    # ==========================================
    if RUN_TRAINING:
        print("\n>>> STAGE 2: Initiating Model Training...")
        train_pipeline()
        print(">>> STAGE 2 Complete.\n")

    # ==========================================
    # STAGE 3 & 4: EVALUATION & VISUALIZATION
    # ==========================================
    if RUN_EVALUATION:
        evaluate_all()
    
if __name__ == "__main__":
    main()