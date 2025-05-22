import os
import re
import torch
import logging
import gc
from datetime import datetime

def clean_text(text):
    """
    Clean text by removing formatting tags and normalizing content
    """
    # Remove common subtitle/formatting tags (expanding the original set)
    text = re.sub(r'{\\fn.*?}', '', text)  # Remove font formatting tags
    text = re.sub(r'{\\fs.*?}', '', text)  # Remove font size tags
    text = re.sub(r'{\\b.*?}', '', text)  # Remove bold tags
    text = re.sub(r'{\\4cH.*?}', '', text)  # Remove color tags
    text = re.sub(r'{\\4aH.*?}', '', text)  # Remove alpha tags
    text = re.sub(r'{\\i.*?}', '', text)  # Remove italic tags
    text = re.sub(r'{\\u.*?}', '', text)  # Remove underline tags
    text = re.sub(r'{\\s.*?}', '', text)  # Remove strike-through tags
    text = re.sub(r'{\\c.*?}', '', text)  # Remove color tags (alternate format)
    text = re.sub(r'{\\pos\(.*?\)}', '', text)  # Remove positioning tags
    text = re.sub(r'{\\an\d+}', '', text)  # Remove alignment tags
    
    # Remove any remaining tags in the format {\\tag}
    text = re.sub(r'{\\.*?}', '', text)
    
    # Remove HTML/XML tags if present
    text = re.sub(r'<.*?>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    # Final cleanup
    text = text.strip()
    
    return text

def write_data(input_text, output_file):

    # processed_path = ('/').join(output_file.split('/')[:-1])
    # try:
    #     os.mkdir(processed_path)
    #     print(f'created new directory in: {processed_path}')
    # except FileExistsError:
    #     print('directory already exist! continuing..')
        
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(input_text)
    print(f"Data saved to {output_file}")
    
    return output_file

def read_data(input_file):
    
    try:
        with open(input_file, 'r') as f:
            text = f.readlines()
            
        return text
    except FileNotFoundError:
        print("Error, file not found! check the filepath again, make sure it exist.")
    


def setup_logging(log_dir='../logs', logname='babylm_nllb_translation'):
    """
    Setup logging configuration
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure timestamp for the log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'{log_dir}/{logname}_{timestamp}.log'
    
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
        return "GPU not available"
    
    memory_info = get_gpu_memory_info()
    memory_str = f"{prefix} GPU Memory - Total: {memory_info['total']:.2f}GB, " \
                f"Reserved: {memory_info['reserved']:.2f}GB, " \
                f"Allocated: {memory_info['allocated']:.2f}GB, " \
                f"Free: {memory_info['free']:.2f}GB"
    logging.info(memory_str)
    
    return memory_str

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
    
    # For NLLB, add source language token
    src_lang = "eng_Latn"  # Default source language is English
    dummy_text = [f"{src_lang} {text}" for text in dummy_text]
    
    # Record initial memory usage
    initial_memory = torch.cuda.memory_allocated(0) / (1024**3)
    
    # Tokenize and process through model
    encoded = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Set target language for generation (default to Indonesian)
    tgt_lang = "ind_Latn"
    forced_tgt_tokens = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    # Generate translation to see memory usage
    with torch.no_grad():
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            forced_bos_token_id=forced_tgt_tokens,
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