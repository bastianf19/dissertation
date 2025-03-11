import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import argparse
import glob
import logging
from datetime import datetime

def setup_logging(log_dir='../logs'):
    """
    Setup logging configuration
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure timestamp for the log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'{log_dir}/babylm_translation_{timestamp}.log'
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return logger

def translate_text_batch(batch, model, tokenizer, device):
    """
    Translate a batch of texts from English to Indonesian
    """
    # Skip empty lines
    batch = [line for line in batch if line.strip()]
    
    if not batch:
        return []
    
    # Tokenize the text
    encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move to GPU if available
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Generate translation
    with torch.no_grad():
        translated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode the translation
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_texts

def translate_file(source_file, target_file, model, tokenizer, device, batch_size=16):
    """
    Translate a file line by line from English to Indonesian
    """
    logging.info(f"Starting translation of: {source_file}")
    
    # Create directory for target file if it doesn't exist
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    
    # Read source file
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        logging.info(f"Read {len(lines)} lines from {source_file}")
    except Exception as e:
        logging.error(f"Error reading {source_file}: {e}")
        return
    
    # Translate in batches
    translated_lines = []
    # total_batches = (len(lines) + batch_size - 1) // batch_size
    
    for i in range(0, len(lines), batch_size):
        try:
            batch = lines[i:i+batch_size]
            batch = [line.strip() for line in batch]
            
            translated_batch = translate_text_batch(batch, model, tokenizer, device)
            translated_lines.extend(translated_batch)
            
            # Log progress periodically
            if (i // batch_size) % 10 == 0 and i > 0:
                logging.info(f"Translated {i + len(batch)} of {len(lines)} lines ({(i + len(batch))/len(lines)*100:.1f}%)")
                
        except Exception as e:
            logging.error(f"Error translating batch at index {i}: {e}")
    
    # Write to target file
    try:
        with open(target_file, 'w', encoding='utf-8') as f:
            for line in translated_lines:
                f.write(line + '\n')
        logging.info(f"Successfully translated {len(translated_lines)} lines to {target_file}")
    except Exception as e:
        logging.error(f"Error writing to {target_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Translate BabyLM dataset from English to Indonesian')
    parser.add_argument('--source_dir', type=str, required=True, help='Root directory containing babylm_data')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to save translated files')
    parser.add_argument('--model_path', type=str, required=True, help='Directory to access MT Model path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for translation (default: 16)')
    parser.add_argument('--subsets', nargs='+', default=['babylm_10M', 'babylm_100M', 'babylm_dev', 'babylm_test'], 
                        help='Subsets to translate (default: all)')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save log files')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    # Log start time and configuration
    logging.info(f"=== Starting BabyLM Dataset Translation Process ===")
    logging.info(f"Source directory: {args.source_dir}")
    logging.info(f"Target directory: {args.target_dir}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Subsets to translate: {args.subsets}")
    
    # Check if directories exist
    source_babylm_dir = os.path.join(args.source_dir, 'babylm_data')
    if not os.path.exists(source_babylm_dir):
        logging.error(f"Source directory {source_babylm_dir} does not exist")
        raise ValueError(f"Source directory {source_babylm_dir} does not exist")
    
    # Create target directory if it doesn't exist
    target_babylm_dir = os.path.join(args.target_dir, 'babylm_data_id')
    os.makedirs(target_babylm_dir, exist_ok=True)
    logging.info(f"Created target directory: {target_babylm_dir}")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load the model and tokenizer
    logging.info("Loading opus-mt-en-id model and tokenizer...")
    try:
        model_name = args.model_path
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        logging.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {e}")
        raise
    
    # Track statistics
    total_files = 0
    translated_files = 0
    
    # Process each subset
    for subset in args.subsets:
        source_subset_dir = os.path.join(source_babylm_dir, subset)
        if not os.path.exists(source_subset_dir):
            logging.warning(f"Subset directory {source_subset_dir} does not exist, skipping...")
            continue
        
        target_subset_dir = os.path.join(target_babylm_dir, subset)
        os.makedirs(target_subset_dir, exist_ok=True)
        logging.info(f"Processing subset: {subset}")
        
        # Get all files in the subset directory
        file_extensions = ['.train', '.dev', '.test']
        files = []
        for ext in file_extensions:
            files.extend(glob.glob(os.path.join(source_subset_dir, f'*{ext}')))
        
        if not files:
            logging.warning(f"No files found in {source_subset_dir}")
            continue
        
        logging.info(f"Found {len(files)} files in {subset}")
        total_files += len(files)
        
        # Translate each file
        for source_file in files:
            base_name = os.path.basename(source_file)
            target_file = os.path.join(target_subset_dir, base_name)
            
            # Translate file
            try:
                translate_file(source_file, target_file, model, tokenizer, device, args.batch_size)
                translated_files += 1
            except Exception as e:
                logging.error(f"Failed to translate {source_file}: {e}")
    
    # Log completion
    logging.info(f"=== Translation process completed ===")
    logging.info(f"Total files found: {total_files}")
    logging.info(f"Successfully translated files: {translated_files}")
    logging.info(f"Translated files saved to {target_babylm_dir}")

if __name__ == "__main__":
    main()