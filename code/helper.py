import os
import re
import gc
import torch
import ctypes
import logging
import unicodedata
from datetime import datetime

WHITELIST = set()

def remove_symbols(text: str, drop_marks: bool = True) -> str:
    """
    Remove every Unicode *Symbol* (category 'S…') and, if `drop_marks`
    is True, every combining *Mark* ('M…').  Whitelist overrides both.
    """
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith('S') and ch not in WHITELIST:
            continue
        if drop_marks and cat.startswith('M'):
            continue
        out.append(ch)
    return ''.join(out)

def clean_text(text):
    """
    Clean text by removing formatting tags and normalizing content
    """
    # ------------------------------------------------------------------
    # 1. Strip common subtitle / styling tags, unicodes
    # ------------------------------------------------------------------
    text = re.sub(r'\{[^}]*\}', '', text)
    text = remove_symbols(text, drop_marks=True)
    text = re.sub(r'[♪♫♩♬♭♮♯∞★☆♥♦♠♣☀☁☂☃☠⚑⚡■]+', '', text)
    text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]+', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'([^\w\s])\1{1,}', r'\1', text)
    
    # ------------------------------------------------------------------
    # 2. Remove repeated words
    # ------------------------------------------------------------------
    # 2-a  Repeated punctuation  (“????” → “?”)
    text = re.sub(r'([?!.,؛—\-])\1{1,}', r'\1', text)

    # 2-b  ≥3 identical words in a row  (“dan dan dan” → “dan”)
    text = re.sub(r'\b(\w+)(?:\s+\1\b){2,}', r'\1', text, flags=re.IGNORECASE)
    
    # ASCII & Unicode letters (e.g. ö, é)
    # text = re.sub(r'([\p{L}])\1{3,}', r'\1\1\1', text, flags=re.UNICODE)

    # 2-c  Extraneous spaces before punctuation  (“benar ?” → “benar?”)
    text = re.sub(r'\s+([?!.,:;])', r'\1', text)

    # ------------------------------------------------------------------
    # 3. Whitespace tidy-up & final trim
    # ------------------------------------------------------------------
    text = re.sub(r'\s+([?!.,])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not re.search(r'[A-Za-zÀ-ȕ0-9]', text):
        return ''
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

def aggressive_memory_cleanup():
    """
    Perform aggressive memory cleanup to force PyTorch to release reserved memory
    """
    if not torch.cuda.is_available():
        return
    
    # Clear all caches
    torch.cuda.empty_cache()
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    # Synchronize CUDA operations
    torch.cuda.synchronize()
    
    # Clear cache again after sync
    torch.cuda.empty_cache()
    
    # Additional memory management for CUDA context
    try:
        # Reset memory stats to help with internal bookkeeping
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except:
        pass
    
    # Force Python garbage collection one more time
    gc.collect()

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
    Log current GPU memory usage with automatic cleanup if memory is high
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
    
    # Perform aggressive cleanup if reserved memory is high (>90% of total)
    reserved_ratio = memory_info['reserved'] / memory_info['total']
    if reserved_ratio > 0.9:
        logging.warning(f"High reserved memory usage ({reserved_ratio:.1%}). Performing aggressive cleanup...")
        aggressive_memory_cleanup()
        
        # Log memory after cleanup
        memory_info_after = get_gpu_memory_info()
        memory_str_after = f"After cleanup - Reserved: {memory_info_after['reserved']:.2f}GB, " \
                          f"Free: {memory_info_after['free']:.2f}GB"
        logging.info(memory_str_after)
    
    return memory_str

def calculate_optimal_batch_size(model, tokenizer, device, target_memory_usage=0.8, max_length=512, initial_batch_size=16):
    """
    Calculate optimal batch size based on available GPU memory
    """
    if not torch.cuda.is_available():
        logging.info("GPU not available, using default batch size")
        return initial_batch_size
    
    # Perform cleanup before calculation
    aggressive_memory_cleanup()
    
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
        generated = model.generate(
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
    
    # Free up memory aggressively
    del input_ids, attention_mask, encoded, generated
    aggressive_memory_cleanup()
    
    # Calculate optimal batch size (with a safety factor)
    available_memory = total_memory * target_memory_usage - initial_memory
    optimal_batch_size = int(available_memory / memory_per_sample)
    
    # Ensure we have a reasonable batch size
    optimal_batch_size = max(1, min(optimal_batch_size, 64))
    
    logging.info(f"Memory per sample: {memory_per_sample:.4f}GB")
    logging.info(f"Recommended batch size for {target_memory_usage*100:.0f}% GPU memory usage: {optimal_batch_size}")
    
    return optimal_batch_size

def memory_efficient_cleanup(batch_interval=10):
    """
    Perform memory cleanup at regular intervals during batch processing
    Returns True if cleanup was performed, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    memory_info = get_gpu_memory_info()
    reserved_ratio = memory_info['reserved'] / memory_info['total']
    
    # More frequent cleanup if memory usage is high
    cleanup_threshold = 0.85 if reserved_ratio > 0.9 else 0.9
    
    if reserved_ratio > cleanup_threshold:
        logging.info(f"Performing scheduled memory cleanup (reserved: {reserved_ratio:.1%})")
        aggressive_memory_cleanup()
        return True
    
    return False

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