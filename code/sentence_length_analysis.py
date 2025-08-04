import os
import re
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

def is_valid_sentence(sentence):
    # Remove punctuation and whitespace
    words = sentence.lower().strip().split()
    
    # Exclude empty sentences
    if not words:
        return False
    
    # Exclude sentences with repeated single word more than 5 times
    # word_counts = defaultdict(int)
    # for word in words:
    #     word_counts[word] += 1
    
    # if any(count > 5 for count in word_counts.values()):
    #     return False
    
    return True

def process_directory(directory):
    sentence_lengths = []
    file_lengths = defaultdict(list)
    
    # Find all .train files
    train_files = [f for f in os.listdir(directory) if f.endswith('.train')]
    
    for filename in train_files:
        filepath = os.path.join(directory, filename)
        print(f"Processing {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if is_valid_sentence(line):
                    words = line.split()
                    sentence_lengths.append(len(words))
                    file_lengths[filename].append(len(words))
    
    return sentence_lengths, file_lengths

def analyze_statistics(data, name):
    print(f"\nStatistics for {name}:")
    print(f"Total valid sentences: {len(data)}")
    print(f"Mean length: {statistics.mean(data):.2f} words")
    print(f"Median length: {statistics.median(data)} words")
    print(f"Standard deviation: {statistics.stdev(data):.2f} words")
    print(f"Minimum length: {min(data)} words")
    print(f"Maximum length: {max(data)} words")
    print(f"25th percentile: {statistics.quantiles(data, n=4)[0]} words")
    print(f"75th percentile: {statistics.quantiles(data, n=4)[2]} words")

def plot_comparison(original_lengths, translated_lengths):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(original_lengths, label='Original', fill=True, alpha=0.3)
    sns.kdeplot(translated_lengths, label='Translated', fill=True, alpha=0.3)
    plt.title('Sentence Length Distribution Comparison')
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('sentence_length_comparison.png')
    plt.close()

def main():
    original_dir = "/home/s2587034/dissertation/dataset/babylm_data/babylm_100M"
    translated_dir = "/home/s2587034/dissertation/dataset/translated/nllb/babylm_data_ind/babylm_100M"
    
    print("Processing original data...")
    original_lengths, original_file_lengths = process_directory(original_dir)
    
    print("\nProcessing translated data...")
    translated_lengths, translated_file_lengths = process_directory(translated_dir)
    
    # Analyze statistics
    analyze_statistics(original_lengths, "Original")
    analyze_statistics(translated_lengths, "Translated")
    
    # Compare distributions
    plot_comparison(original_lengths, translated_lengths)
    
    # Save detailed statistics to file
    with open('sentence_length_stats.txt', 'w', encoding='utf-8') as f:
        f.write("Original Data Statistics:\n")
        f.write(f"Total valid sentences: {len(original_lengths)}\n")
        f.write(f"Mean length: {statistics.mean(original_lengths):.2f} words\n")
        f.write(f"Median length: {statistics.median(original_lengths)} words\n")
        f.write(f"Standard deviation: {statistics.stdev(original_lengths):.2f} words\n")
        
        f.write("\nTranslated Data Statistics:\n")
        f.write(f"Total valid sentences: {len(translated_lengths)}\n")
        f.write(f"Mean length: {statistics.mean(translated_lengths):.2f} words\n")
        f.write(f"Median length: {statistics.median(translated_lengths)} words\n")
        f.write(f"Standard deviation: {statistics.stdev(translated_lengths):.2f} words\n")

if __name__ == "__main__":
    main()
