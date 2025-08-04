import argparse
import numpy as np
import os
import re
from pathlib import Path
from transformers import AutoTokenizer

def calculate_tokenizer_fertility(text, tokenizer):
    """
    Calculate various fertility metrics for a given text and tokenizer.
    
    Args:
        text (str): Input text to analyze
        tokenizer: Tokenizer object
        
    Returns:
        dict: Dictionary containing various fertility metrics
    """
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    
    # Calculate basic metrics
    num_tokens = len(tokens)
    num_chars = len(text)
    num_words = len(text.split())
    
    # Token-character fertility
    token_char_fertility = num_tokens / num_chars if num_chars > 0 else 0
    
    # Token-word fertility
    token_word_fertility = num_tokens / num_words if num_words > 0 else 0
    
    # Token length statistics
    token_lengths = [len(token) for token in tokens]
    avg_token_length = np.mean(token_lengths) if token_lengths else 0
    std_token_length = np.std(token_lengths) if token_lengths else 0
    
    return {
        'num_tokens': num_tokens,
        'num_chars': num_chars,
        'num_words': num_words,
        'token_char_fertility': token_char_fertility,
        'token_word_fertility': token_word_fertility,
        'avg_token_length': avg_token_length,
        'std_token_length': std_token_length,
        'token_lengths': token_lengths
    }

def analyze_tokenizer_fertility(texts, tokenizer):
    """
    Analyze tokenizer fertility across multiple texts.
    
    Args:
        texts (list of str): List of texts to analyze
        tokenizer: Tokenizer object
        
    Returns:
        dict: Aggregated statistics across all texts
    """
    all_metrics = []
    
    for text in texts:
        metrics = calculate_tokenizer_fertility(text, tokenizer)
        all_metrics.append(metrics)
    
    # Calculate aggregated statistics
    aggregated = {
        'avg_token_char_fertility': np.mean([m['token_char_fertility'] for m in all_metrics]),
        'avg_token_word_fertility': np.mean([m['token_word_fertility'] for m in all_metrics]),
        'avg_token_length': np.mean([m['avg_token_length'] for m in all_metrics]),
        'std_token_length': np.mean([m['std_token_length'] for m in all_metrics]),
        'total_tokens': sum(m['num_tokens'] for m in all_metrics),
        'total_chars': sum(m['num_chars'] for m in all_metrics),
        'total_words': sum(m['num_words'] for m in all_metrics)
    }
    
    return aggregated

def clean_text(text):
    """
    Clean text by removing special symbols and normalizing whitespace.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text with special symbols removed and normalized whitespace
    """
    # Remove musical notes and other special symbols
    text = re.sub(r'[♪♫♩♬♭♮♯\u200b\u200c\u200d\u2060\u3000\u00a0]+', ' ', text)
    
    # Remove other non-alphanumeric characters except basic punctuation and letters
    text = re.sub(r'[^\w\s.,!?;:()"\'-]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

def process_directory(directory_path):
    """
    Process all .train files in a directory and return their contents.
    
    Args:
        directory_path (str): Path to directory containing .train files
        
    Returns:
        dict: Dictionary mapping file paths to their cleaned contents
    """
    train_files = {}
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return train_files
    
    # List all files in the directory and filter for .train files
    try:
        all_files = os.listdir(directory_path)
        train_file_paths = [f for f in all_files if f.endswith('.train')]
        
        for filename in train_file_paths:
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.readlines()
                    # Clean each line and filter out empty lines
                    cleaned_content = [clean_line for line in content 
                                     if (clean_line := clean_text(line.strip()))]
                    train_files[file_path] = cleaned_content
            except Exception as e:
                print(f"Warning: Could not process file {file_path}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error reading directory {directory_path}: {str(e)}")
    
    return train_files

def main():
    parser = argparse.ArgumentParser(description='Tokenizer Fertility Analysis')
    parser.add_argument('--model_path', type=str, default="/home/s2587034/dissertation/models/indo_tokenizer",
                      help='Path to the tokenizer directory')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to directory containing .train files')
    args = parser.parse_args()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print(f"\nSearching for .train files in directory: {args.data_dir}")
    if not os.path.exists(args.data_dir):
        print(f"Error: The specified directory does not exist: {args.data_dir}")
        return
        
    train_files = process_directory(args.data_dir)
    
    if not train_files:
        print("\nNo .train files found in the specified directory!")
        print("The directory contains the following files:")
        for f in os.listdir(args.data_dir):
            print(f"- {f}")
        return
    
    print(f"\nFound {len(train_files)} .train files. Starting analysis...")
    for file_path in train_files:
        print(f"- {file_path}")
    
    # Analyze each file and collect results
    file_results = {}
    for file_path, texts in train_files.items():
        print(f"\nAnalyzing {os.path.basename(file_path)}...")
        results = analyze_tokenizer_fertility(texts, tokenizer)
        file_results[file_path] = results
        print(f"  Tokens per character: {results['avg_token_char_fertility']:.4f}")
        print(f"  Tokens per word: {results['avg_token_word_fertility']:.4f}")
        print(f"  Avg token length: {results['avg_token_length']:.2f} chars")
    
    # Calculate and print overall statistics
    if file_results:
        print("\n=== Overall Statistics ===")
        total_tokens = sum(r['total_tokens'] for r in file_results.values())
        total_chars = sum(r['total_chars'] for r in file_results.values())
        total_words = sum(r['total_words'] for r in file_results.values())
        
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total characters: {total_chars:,}")
        print(f"Total words: {total_words:,}")
        print(f"Overall tokens/character: {sum(r['total_tokens'] for r in file_results.values()) / total_chars:.4f}" if total_chars > 0 else "No characters found")
        print(f"Overall tokens/word: {sum(r['total_tokens'] for r in file_results.values()) / total_words:.4f}" if total_words > 0 else "No words found")

if __name__ == '__main__':
    main()
