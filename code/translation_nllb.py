import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
import glob
import logging
import gc
from helper import setup_logging, log_gpu_memory, calculate_optimal_batch_size, get_pending_files

def translate_text_batch(batch, model, tokenizer, device, src_lang="eng_Latn", tgt_lang="ind_Latn", max_length=768):
    """
    Translate a batch of texts using NLLB model with dynamic length handling,
    preserving empty lines and special characters
    """
    # Create a mask to track which items are empty lines
    empty_mask = [line.strip() == "" for line in batch]
    
    # Only process non-empty lines
    non_empty_batch = [line for line in batch if line.strip()]
    
    # If all lines are empty, return the original batch
    if not non_empty_batch:
        return batch
    
    # Add source language token to each input
    non_empty_batch = [f"{src_lang} {text}" for text in non_empty_batch]
    
    # Determine if we need to use longer sequence length
    long_sequences = any(len(tokenizer.encode(text)) > 512 for text in non_empty_batch)
    
    # Use a larger max_length for long sequences
    actual_max_length = max_length if long_sequences else 512
    
    # Tokenize the text
    encoded = tokenizer(non_empty_batch, return_tensors="pt", padding=True, truncation=True, max_length=actual_max_length)
    
    # Move to GPU if available
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Set the target language for generation
    forced_tgt_tokens = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    # Generate translation
    with torch.no_grad():
        translated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=actual_max_length,
            num_beams=4,
            forced_bos_token_id=forced_tgt_tokens,
            early_stopping=True
        )
    
    # Decode the translation
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    # Free up memory explicitly
    del input_ids, attention_mask, encoded, translated
    
    # Reconstruct the batch with empty lines preserved
    result = []
    non_empty_index = 0
    for is_empty in empty_mask:
        if is_empty:
            result.append("")  # Keep empty lines as empty
        else:
            result.append(translated_texts[non_empty_index])
            non_empty_index += 1
    
    return result

def translate_file(source_file, target_file, model, tokenizer, device, src_lang="eng_Latn", tgt_lang="ind_Latn", batch_size=16, max_length=768):
    """
    Translate a file line by line using NLLB with adaptive batching,
    preserving all newlines and formatting
    """
    logging.info(f"Starting translation of: {source_file}")
    log_gpu_memory("Before translation:")
    
    # Create directory for target file if it doesn't exist
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    
    # Read source file, preserving all whitespace and newlines
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        logging.info(f"Read {len(lines)} lines from {source_file}")
    except Exception as e:
        logging.error(f"Error reading {source_file}: {e}")
        return
    
    # Calculate token statistics for non-empty lines only
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines:
        token_lengths = [len(tokenizer.encode(line.strip())) for line in non_empty_lines]
        avg_length = sum(token_lengths) / len(token_lengths)
        max_tokens = max(token_lengths)
    else:
        avg_length = 0
        max_tokens = 0
    
    logging.info(f"Average tokens per non-empty line: {avg_length:.1f}, Maximum tokens: {max_tokens}")
    
    # Adjust batch size for very long sequences
    adaptive_batch_size = batch_size
    if max_tokens > 256:
        scale_factor = min(1.0, 256 / max(1, avg_length))
        adaptive_batch_size = max(1, int(batch_size * scale_factor))
        logging.info(f"Adjusted batch size to {adaptive_batch_size} due to long sequences")
    
    # Translate in batches
    translated_lines = []
    
    # Create progress bar
    progress_bar = range(0, len(lines), adaptive_batch_size)
    
    for i in progress_bar:
        try:
            # Get batch - keep original lines intact, including newlines
            batch = lines[i:i+adaptive_batch_size]
            
            # Strip newlines only for translation, but preserve them for later
            batch_for_translation = [line.rstrip('\n') for line in batch]
            
            translated_batch = translate_text_batch(batch_for_translation, model, tokenizer, device, src_lang, tgt_lang, max_length)
            
            # Add back newlines that were in the original
            translated_batch_with_newlines = [line + ('\n' if i < len(batch) and batch[i].endswith('\n') else '') 
                                               for i, line in enumerate(translated_batch)]
            
            translated_lines.extend(translated_batch_with_newlines)
            
            # Update progress description
            logging.info(f"Translated {i + len(batch)}/{len(lines)} lines, {(i + len(batch)) / len(lines):.2%}; {log_gpu_memory('After batch:')}")
            
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
                batch_for_translation = [line.rstrip('\n') for line in batch]
                translated_batch = translate_text_batch(batch_for_translation, model, tokenizer, device, src_lang, tgt_lang, max_length)
                
                # Add back newlines
                translated_batch_with_newlines = [line + ('\n' if i < len(batch) and batch[i].endswith('\n') else '') 
                                                 for i, line in enumerate(translated_batch)]
                
                translated_lines.extend(translated_batch_with_newlines)
                
                # Continue with smaller batch size
                adaptive_batch_size = new_batch_size
            else:
                logging.error(f"Error translating batch at index {i}: {e}")
        except Exception as e:
            logging.error(f"Error translating batch at index {i}: {e}")
    
    # Write to target file
    try:
        with open(target_file, 'w', encoding='utf-8') as f:
            f.writelines(translated_lines)
        logging.info(f"Successfully translated {len(translated_lines)} lines to {target_file}")
    except Exception as e:
        logging.error(f"Error writing to {target_file}: {e}")
    
    log_gpu_memory("After translation:")


def main():
    parser = argparse.ArgumentParser(description='Translate BabyLM dataset using NLLB model')
    parser.add_argument('--source_dir', type=str, required=True, help='Root directory containing babylm_data')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to save translated files')
    parser.add_argument('--model_path', type=str, default='facebook/nllb-200-distilled-600M', 
                      help='NLLB model to use (default: facebook/nllb-200-distilled-600M)')
    parser.add_argument('--src_lang', type=str, default='eng_Latn', help='Source language code (default: eng_Latn)')
    parser.add_argument('--tgt_lang', type=str, default='ind_Latn', help='Target language code (default: ind_Latn)')
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
    logging.info(f"=== Starting BabyLM Dataset Translation Process with NLLB ===")
    logging.info(f"Source directory: {args.source_dir}")
    logging.info(f"Target directory: {args.target_dir}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Source language: {args.src_lang}")
    logging.info(f"Target language: {args.tgt_lang}")
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
    target_suffix = f"{args.tgt_lang.split('_')[0]}"  # Extract language code (e.g., ind from ind_Latn)
    target_babylm_dir = os.path.join(args.target_dir, f'babylm_data_{target_suffix}')
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
    
    # Load the NLLB model and tokenizer
    logging.info(f"Loading NLLB model and tokenizer from {args.model_path}...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        logging.info("NLLB model and tokenizer loaded successfully")
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
            if 'open_subtitles.train' in source_file:
                logging.info("open_subtitles too large! skipping for now..")
                continue
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
                    src_lang=args.src_lang,
                    tgt_lang=args.tgt_lang,
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