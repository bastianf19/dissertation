import os
import torch
from transformers import MarianMTModel, MarianTokenizer
import argparse
import glob
import logging
from datetime import datetime
import gc
from tqdm import tqdm

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
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
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

def get_gpu_memory_info():
    """
    Get current GPU memory usage
    """
    if not torch.cuda.is_available():
        return "GPU not available"
    
    # Get current GPU memory usage
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)  # GB
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
    free_memory = total_memory - reserved_memory
    
    memory_info = {
        "total": total_memory,
        "reserved": reserved_memory,
        "allocated": allocated_memory,
        "free": free_memory
    }
    
    return memory_info

def log_gpu_memory(prefix=""):
    """
    Log current GPU memory usage
    """
    if not torch.cuda.is_available():
        logging.info("GPU not available")
        return
    
    memory_info = get_gpu_memory_info()
    logging.info(f"{prefix} GPU Memory - Total: {memory_info['total']:.2f}GB, "
                f"Reserved: {memory_info['reserved']:.2f}GB, "
                f"Allocated: {memory_info['allocated']:.2f}GB, "
                f"Free: {memory_info['free']:.2f}GB")

def calculate_optimal_batch_size(model, tokenizer, device, target_memory_usage=0.8, max_length=512, initial_batch_size=16):
    """
    Calculate optimal batch size based on available GPU memory
    """
    if not torch.cuda.is_available():
        logging.info("GPU not available, using default batch size")
        return initial_batch_size
    
    memory_info = get_gpu_memory_info()
    total_memory = memory_info["total"]
    
    # Start with a small batch and dummy input
    batch_size = 2
    dummy_text = ["This is a dummy text for batch size calculation."] * batch_size
    
    # Record initial memory usage
    initial_memory = torch.cuda.memory_allocated(0) / (1024**3)
    
    # Tokenize and process through model
    encoded = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Generate translation to see memory usage
    with torch.no_grad():
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    # Calculate memory used per sample
    current_memory = torch.cuda.memory_allocated(0) / (1024**3)
    memory_per_sample = (current_memory - initial_memory) / batch_size
    
    # Free up memory
    del input_ids, attention_mask, encoded
    torch.cuda.empty_cache()
    gc.collect()
    
    # Calculate optimal batch size (with a safety factor)
    available_memory = total_memory * target_memory_usage - initial_memory
    optimal_batch_size = int(available_memory / memory_per_sample)
    
    # Ensure we have a reasonable batch size
    optimal_batch_size = max(1, min(optimal_batch_size, 64))
    
    logging.info(f"Memory per sample: {memory_per_sample:.4f}GB")
    logging.info(f"Recommended batch size for {target_memory_usage*100:.0f}% GPU memory usage: {optimal_batch_size}")
    
    return optimal_batch_size

def translate_text_batch(batch, model, tokenizer, device, max_length=768):
    """
    Translate a batch of texts from English to Indonesian with dynamic length handling
    """
    # Skip empty lines
    batch = [line for line in batch if line.strip()]
    
    if not batch:
        return []
    
    # Determine if we need to split long sequences
    long_sequences = any(len(tokenizer.encode(text)) > 512 for text in batch)
    
    # Use a larger max_length for long sequences
    actual_max_length = max_length if long_sequences else 512
    
    # Tokenize the text
    encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=actual_max_length)
    
    # Move to GPU if available
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Generate translation
    with torch.no_grad():
        translated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=actual_max_length,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode the translation
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    # Free up memory explicitly
    del input_ids, attention_mask, encoded, translated
    
    return translated_texts

def translate_file(source_file, target_file, model, tokenizer, device, batch_size=16, max_length=768):
    """
    Translate a file line by line from English to Indonesian with adaptive batching
    """
    logging.info(f"Starting translation of: {source_file}")
    log_gpu_memory("Before translation:")
    
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
    
    # Analyze sentence lengths to determine if we need smaller batches
    token_lengths = [len(tokenizer.encode(line.strip())) for line in lines if line.strip()]
    avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0
    
    logging.info(f"Average tokens per line: {avg_length:.1f}, Maximum tokens: {max_tokens}")
    
    # Adjust batch size for very long sequences
    adaptive_batch_size = batch_size
    if max_tokens > 256:
        scale_factor = min(1.0, 256 / avg_length)
        adaptive_batch_size = max(1, int(batch_size * scale_factor))
        logging.info(f"Adjusted batch size to {adaptive_batch_size} due to long sequences")
    
    # Translate in batches
    translated_lines = []
    
    # Use tqdm for progress bar
    progress_bar = range(0, len(lines), adaptive_batch_size)
    
    for i in progress_bar:
        try:
            batch = lines[i:i+adaptive_batch_size]
            batch = [line.strip() for line in batch]
            
            translated_batch = translate_text_batch(batch, model, tokenizer, device, max_length)
            translated_lines.extend(translated_batch)
            
            # Update progress description
            logging.info(f"Translated {i + len(batch)}/{len(lines)} lines, {i + len(batch) / len(lines)} %; {log_gpu_memory(f'After {i + len(batch)} lines:')}")
            
            # Check memory usage periodically
            if (i // adaptive_batch_size) % 50 == 0 and i > 0:
                log_gpu_memory(f"After {i + len(batch)} lines:")
                # Explicitly clean up any unused memory
                torch.cuda.empty_cache()
                gc.collect()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Handle OOM error by reducing batch size and retrying
                logging.warning(f"OOM error. Reducing batch size and retrying.")
                torch.cuda.empty_cache()
                gc.collect()
                
                # Try with half the batch size
                new_batch_size = max(1, adaptive_batch_size // 2)
                logging.info(f"Retrying with batch size {new_batch_size}")
                
                # Retry with smaller batch
                batch = lines[i:i+new_batch_size]
                batch = [line.strip() for line in batch]
                translated_batch = translate_text_batch(batch, model, tokenizer, device, max_length)
                translated_lines.extend(translated_batch)
                
                # Continue with smaller batch size
                adaptive_batch_size = new_batch_size
            else:
                logging.error(f"Error translating batch at index {i}: {e}")
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
    
    log_gpu_memory("After translation:")

def get_pending_files(source_files, target_dir):
    """
    Get files that still need to be processed (don't exist in target directory)
    """
    pending_files = []
    skipped_files = []
    
    for source_file in source_files:
        base_name = os.path.basename(source_file)
        target_file = os.path.join(target_dir, base_name)
        
        if os.path.exists(target_file):
            # Check if the file is empty or corrupt
            try:
                with open(target_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:  # Empty file
                    logging.info(f"Target file {target_file} exists but is empty. Will reprocess.")
                    pending_files.append(source_file)
                else:
                    skipped_files.append(source_file)
            except Exception:
                # If there's an error reading the file, assume it's corrupt and reprocess
                logging.warning(f"Error reading target file {target_file}. Will reprocess.")
                pending_files.append(source_file)
        else:
            pending_files.append(source_file)
    
    return pending_files, skipped_files

def main():
    parser = argparse.ArgumentParser(description='Translate BabyLM dataset from English to Indonesian')
    parser.add_argument('--source_dir', type=str, required=True, help='Root directory containing babylm_data')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to save translated files')
    parser.add_argument('--model_path', type=str, required=True, help='Directory to access MT Model path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for translation (default: 16)')
    parser.add_argument('--max_length', type=int, default=768, help='Maximum sequence length for long sentences (default: 768)')
    parser.add_argument('--auto_batch_size', action='store_true', help='Automatically determine optimal batch size')
    parser.add_argument('--subsets', nargs='+', default=['babylm_10M', 'babylm_100M', 'babylm_dev', 'babylm_test'], 
                        help='Subsets to translate (default: all)')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save log files')
    parser.add_argument('--memory_target', type=float, default=0.85, help='Target GPU memory usage (0.0-1.0)')
    parser.add_argument('--force_all', action='store_true', help='Force translation of all files even if they exist in target')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    # Log start time and configuration
    logging.info(f"=== Starting BabyLM Dataset Translation Process ===")
    logging.info(f"Source directory: {args.source_dir}")
    logging.info(f"Target directory: {args.target_dir}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Initial batch size: {args.batch_size}")
    logging.info(f"Max sequence length: {args.max_length}")
    logging.info(f"Auto batch size: {args.auto_batch_size}")
    logging.info(f"Memory usage target: {args.memory_target * 100:.0f}%")
    logging.info(f"Subsets to translate: {args.subsets}")
    logging.info(f"Force translate all files: {args.force_all}")
    
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
    
    # Log GPU information if available
    if torch.cuda.is_available():
        log_gpu_memory("Initial")
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"GPU: {gpu_name}")
    
    # Load the model and tokenizer
    logging.info("Loading opus-mt-en-id model and tokenizer...")
    try:
        model_name = args.model_path
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        logging.info("Model and tokenizer loaded successfully")
        log_gpu_memory("After model loading")
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {e}")
        raise
    
    # Determine optimal batch size if requested
    batch_size = args.batch_size
    if args.auto_batch_size and torch.cuda.is_available():
        batch_size = calculate_optimal_batch_size(
            model, tokenizer, device, 
            target_memory_usage=args.memory_target,
            max_length=args.max_length,
            initial_batch_size=args.batch_size
        )
        logging.info(f"Using auto-determined batch size: {batch_size}")
    
    # Track statistics
    total_files = 0
    translated_files = 0
    skipped_files = 0
    
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
        all_files = []
        for ext in file_extensions:
            all_files.extend(glob.glob(os.path.join(source_subset_dir, f'*{ext}')))
        
        if not all_files:
            logging.warning(f"No files found in {source_subset_dir}")
            continue
        
        logging.info(f"Found {len(all_files)} files in {subset}")
        total_files += len(all_files)
        
        # Get list of files that need to be processed
        if args.force_all:
            pending_files = all_files
            subset_skipped_files = []
            logging.info(f"Forced processing of all {len(pending_files)} files in {subset}")
        else:
            pending_files, subset_skipped_files = get_pending_files(all_files, target_subset_dir)
            logging.info(f"Found {len(pending_files)} files to process and {len(subset_skipped_files)} files already processed in {subset}")
        
        skipped_files += len(subset_skipped_files)
        
        # Skip if no files need processing
        if not pending_files:
            logging.info(f"All files in {subset} are already processed. Skipping...")
            continue
        
        # Translate pending files
        for source_file in pending_files:
            base_name = os.path.basename(source_file)
            target_file = os.path.join(target_subset_dir, base_name)
            
            # Translate file
            try:
                translate_file(
                    source_file, 
                    target_file, 
                    model, 
                    tokenizer, 
                    device, 
                    batch_size=batch_size,
                    max_length=args.max_length
                )
                translated_files += 1
                
                # Clean up after each file
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                logging.error(f"Failed to translate {source_file}: {e}")
    
    # Log completion
    logging.info(f"=== Translation process completed ===")
    logging.info(f"Total files found: {total_files}")
    logging.info(f"Files already processed and skipped: {skipped_files}")
    logging.info(f"Files translated in this run: {translated_files}")
    logging.info(f"Translated files saved to {target_babylm_dir}")
    log_gpu_memory("Final")

if __name__ == "__main__":
    main()