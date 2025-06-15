from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os

def train_indonesian_tokenizer():
    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE())
    
    # Use whitespace pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    
    # Define special tokens
    special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    
    # Create a trainer
    trainer = BpeTrainer(
        vocab_size=50000,  # Increased based on larger dataset size
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Get all text files from the dataset directory
    dataset_paths = []
    base_dir = "/home/s2587034/dissertation/dataset/clean_data/nllb/babylm_data_ind"
    
    # Add files from all relevant directories
    directories = [
        "babylm_10M",
        "babylm_100M",
        "babylm_dev",
        "babylm_test"
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".train") or file.endswith(".dev") or file.endswith(".test"):
                        dataset_paths.append(os.path.join(root, file))
    
    print(f"Training tokenizer on {len(dataset_paths)} files")
    print("Files included:")
    for path in dataset_paths:
        print(path)
    
    # Train the tokenizer
    tokenizer.train(dataset_paths, trainer)
    
    # Add special tokens
    tokenizer.add_special_tokens(special_tokens)
    
    # Add post-processing
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", 0),
            ("</s>", 1)
        ]
    )
    
    # Save the tokenizer
    output_dir = "/home/s2587034/dissertation/models/custom_tokenizer"
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    print(f"Tokenizer saved to {output_dir}")

if __name__ == "__main__":
    train_indonesian_tokenizer()
