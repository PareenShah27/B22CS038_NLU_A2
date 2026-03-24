"""
Handles Task-1: Dataset Preparation

Responsibilities:
1. Reads raw text files from the 'data/raw' directory.
2. Cleans text: removes boilerplate, formatting artifacts, non-English text, and punctuation.
3. Preprocesses: Tokenization and Lowercasing.
4. Generates Statistics: Total docs, tokens, and vocabulary size.
5. Visualization: Generates and saves a Word Cloud of the most frequent words.
"""

import os
import re
import spacy
from typing import List
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import PyPDF2
import config

NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # Load spaCy model for tokenization
NLP.max_length = 20000000 # Increase max length to handle large documents

def extract_from_pdf(file_path: str) -> str:
    """
    Extracts text content from a PDF file.
    
    Args:
        file_path (str): The path to the PDF file.
    Returns:
        str: The concatenated raw text from all pages.
    """

    raw_text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted + " "
    except Exception as e:
        print(f"[Error] Could not read PDF {file_path}. Error: {e}")
    
    return raw_text


def clean_text(raw_text: str) -> List[str]:
    """
    Cleans raw text and tokenizes it using spaCy.
    
    Args:
        raw_text (str): The raw string content of a document.
    Returns:
        list: A list of cleaned, lowercased string tokens.
    """

    text = re.sub(r'<[^>]+>', ' ', raw_text) # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text) # Remove URLs
    text = re.sub(r'\S+@\S+', '', text) # Remove email addresses
    text = re.sub(r'page \d+ of \d+', '', text) # Remove page numbers
    text = re.sub(r'copyright © \d+', '', text) # Remove copyright notices
    
    doc = NLP(text)

    tokens = []
    for token in doc:
        # Filter out punctuation, whitespace, and non-alphabetic tokens
        if not token.is_stop and not token.is_punct and token.is_alpha:
            tokens.append(token.lower_)


    return tokens

def gen_wordcloud(tokens: List[str], save_path: str):
    """
    Generates and saves a Word Cloud visualization of the most frequent words.
    
    Args:
        tokens (list): A list of string tokens from the corpus.
        save_path (str): The file path to save the generated Word Cloud image.
    """
    text_content = " ".join(tokens)
    
    # Generate the cloud
    wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text_content)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Most Frequent Words in IIT Jodhpur Corpus")

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Word Cloud saved to: {save_path}") 
    plt.close()

def process_data():
    """
    Main driver function for Task-1.
    Iterates through raw files, cleans them, builds the corpus, and reports stats.
    """
    print("---Starting Data Preprocessing---")

    # Intitialize Counters
    total_docs = 0
    all_tokens = []    # Stores every single token found
    cleaned_lines = [] # Stores processed lines to write to file

    # Check if raw directory exists
    if not os.path.exists(config.DATA_RAW_DIR):
        print(f"Error: Raw data directory '{config.DATA_RAW_DIR}' does not exist.")
        return
    
    # Iterate over all files in the raw data directory
    for filename in os.listdir(config.DATA_RAW_DIR):
        file_path = os.path.join(config.DATA_RAW_DIR, filename)

        if filename.lower().endswith(".pdf"):
            try:
                raw_content = extract_from_pdf(file_path)
                
                if raw_content:
                    doc_tokens = clean_text(raw_content)

                    if doc_tokens:
                        total_docs += 1
                        all_tokens.extend(doc_tokens)
                        cleaned_lines.append(" ".join(doc_tokens))
            except Exception as e:
                print(f"Warning: Could not process PDF {filename}. Error: {e}")
    
    # --- Save Cleaned Corpus ---
    os.makedirs(os.path.dirname(config.DATA_CLEANED_PATH), exist_ok=True)
    with open(config.DATA_CLEANED_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(cleaned_lines))
    
    print(f"[IO] Cleaned corpus saved to: {config.DATA_CLEANED_PATH}")

    # --- Generate Statistics ---
    unique_tokens = set(all_tokens)
    vocab_size = len(unique_tokens)
    tokens_cnt = len(all_tokens)

    print("\n--- Dataset Statistics ---")
    print(f"1. Total Number of Documents: {total_docs}")
    print(f"2. Total Number of Tokens:    {tokens_cnt}")
    print(f"3. Vocabulary Size:           {vocab_size}")

    # --- Generate Word Cloud ---
    if tokens_cnt > 0:
        os.makedirs(config.PLOTS_DIR, exist_ok=True)
        wordcloud_path = os.path.join(config.PLOTS_DIR, "wordcloud.png")
        gen_wordcloud(all_tokens, wordcloud_path)
    else:
        print("Warning: No tokens found. Skipping Word Cloud generation.")
    

if __name__ == "__main__":
    # For Independent testing of the preprocessor
    process_data()

