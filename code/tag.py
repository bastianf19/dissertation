# For importing utils
import sys
sys.path.append("..")

import pytest
import glob
import os
import argparse
import stanza
import json
import torch
import logging
import time
from helper import setup_logging

# Set up logging
setup_logging(logname='tag_processing')


test_all_files = sorted(glob.glob("dataset/translated/nllb/babylm_data_ind/babylm_*/*"))
test_original_files = [f for f in test_all_files if ".json" not in f]
test_json_files = [f for f in test_all_files if "_parsed.json" in f]
test_cases = list(zip(test_original_files, test_json_files))


def log_gpu_memory():
    """Log current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # Convert to GB
        logging.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        logging.info("GPU Memory - CUDA not available")


def clear_gpu_cache():
    """Clear GPU cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logging.info("GPU cache cleared")


@pytest.mark.parametrize("original_file, json_file", test_cases)
def test_equivalent_lines(original_file, json_file):
    # Read lines of file and remove all whitespace
    with open(original_file) as f:
        original_data = "".join(f.readlines())
    original_data = "".join(original_data.split())

    with open(json_file) as f:
        json_lines = json.load(f)
    json_data = ""
    for line in json_lines:
        for sent in line["sent_annotations"]:
            json_data += sent["sent_text"]
    json_data = "".join(json_data.split())

    # Test equivalence
    assert (original_data == json_data)


def check_output_exists(file_path, parse_mode):
    """Check if output file already exists."""
    ext = '_parsed.json' if parse_mode else '.json'
    json_filename = os.path.splitext(file_path)[0] + ext
    return os.path.exists(json_filename)


def filter_files_by_subset(file_paths, subsets):
    """Filter file paths to include only specified subsets."""
    if not subsets:
        return file_paths
    
    filtered_files = []
    valid_subsets = ['babylm_10M', 'babylm_100M', 'babylm_dev', 'babylm_test']
    
    invalid_subsets = [s for s in subsets if s not in valid_subsets]
    if invalid_subsets:
        logging.warning(f"Invalid subset names: {invalid_subsets}")
        logging.info(f"Valid subsets are: {valid_subsets}")
    
    for file_path in file_paths:
        for subset in subsets:
            if subset in valid_subsets and f"/{subset}/" in file_path:
                filtered_files.append(file_path)
                break
    
    return filtered_files


def list_available_subsets(file_paths):
    """List all available dataset subsets."""
    subsets = set()
    for file_path in file_paths:
        path_parts = file_path.split('/')
        for part in path_parts:
            if part.startswith('babylm_'):
                subsets.add(part)
    
    logging.info("Available dataset subsets:")
    for subset in sorted(subsets):
        count = len([f for f in file_paths if f"/{subset}/" in f])
        logging.info(f"  {subset}: {count} files")
    
    return sorted(subsets)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Tag BabyLM dataset',
        description='Tag BabyLM dataset using Stanza')
    parser.add_argument('path', type=str, nargs='+',
                        help="Path to file(s) (can include glob patterns)")
    parser.add_argument('-p', '--parse', action='store_true',
                        help="Include constituency parse")
    parser.add_argument('--max-seq-length', type=int, default=512,
                        help="Maximum sequence length for processing (default: 512)")
    parser.add_argument('--cpu-only', action='store_true',
                        help="Force CPU-only processing to avoid cuDNN issues")
    parser.add_argument('--force', action='store_true',
                        help="Force reprocessing even if output files already exist")
    parser.add_argument('--subset', type=str, nargs='+',
                        help="Process only specific dataset subsets (e.g., babylm_10M)")
    parser.add_argument('--list-subsets', action='store_true',
                        help="List available dataset subsets and exit")
    parser.add_argument('--filename', type=str,
                        help="Filename to process")
    parser.add_argument('--batch-size', type=int, default=2000,
                        help="Number of lines to process in a single batch (default: 2000)")

    args = parser.parse_args()
    
    logging.info(f'Use parsing: {args.parse}')
    logging.info(f'Batch size: {args.batch_size}')

    if args.list_subsets:
        file_paths = [f for pattern in args.path for f in glob.glob(pattern)]
        if file_paths:
            list_available_subsets(file_paths)
        else:
            logging.error("No files found to analyze subsets")
        sys.exit(0)

    torch.backends.cudnn.enabled = not args.cpu_only
    torch.backends.cudnn.benchmark = False

    log_gpu_memory()

    logging.info("Initializing Stanza NLP pipeline...")
    processors = 'tokenize,pos,lemma,constituency' if args.parse else 'tokenize,pos,lemma'
    
    processor_config = {
        'tokenize': {'batch_size': 1000},
        'pos': {'batch_size': 1000, 'bert_finetune': False},
        'lemma': {'batch_size': 1000},
        'constituency': {'batch_size': 200} if args.parse else {}
    }

    try:
        nlp = stanza.Pipeline(
            # lang='en', # for the original babyLM
            lang='id',
            processors=processors,
            use_gpu=not args.cpu_only,
            download_method=None,
            tokenize_no_ssplit=True,
            **processor_config
        )
        logging.info("Stanza pipeline initialized successfully on GPU.")
    except Exception as e:
        logging.error(f"Failed to initialize GPU pipeline: {e}. Falling back to CPU.")
        nlp = stanza.Pipeline(
            # lang='en', # for the original babyLM
            lang='id',
            processors=processors,
            use_gpu=False,
            download_method=None,
            tokenize_no_ssplit=True
        )

    BATCH_SIZE = args.batch_size
    
    file_paths = [f for pattern in args.path for f in glob.glob(pattern)]
    if not file_paths:
        logging.error("No files found to process")
        sys.exit(1)
    
    logging.info(f"Found {len(file_paths)} total files")
    
    if args.subset:
        file_paths = filter_files_by_subset(file_paths, args.subset)
        logging.info(f"Filtered to {len(file_paths)} files for subsets: {args.subset}")
        if not file_paths:
            logging.error("No files found for the specified subsets. Use --list-subsets to see options.")
            sys.exit(1)
    
    if args.filename:
        file_paths = [fp for fp in file_paths if os.path.basename(fp) == args.filename]
        if not file_paths:
            logging.error(f"Specified filename '{args.filename}' not found.")
            sys.exit(1)

    if not args.force:
        original_count = len(file_paths)
        file_paths = [fp for fp in file_paths if not check_output_exists(fp, args.parse) and not fp.endswith('.json')]
        skipped_count = original_count - len(file_paths)
        if skipped_count > 0:
            logging.info(f"Skipped {skipped_count} files that were already processed or are JSON files.")
    else:
        file_paths = [fp for fp in file_paths if not fp.endswith('.json')]

    if not file_paths:
        logging.info("All specified files have already been processed. Use --force to reprocess.")
        sys.exit(0)
    
    logging.info(f"Beginning processing for {len(file_paths)} files...")
    total_start_time = time.time()
    
    for file_idx, file_path in enumerate(file_paths, 1):
        start_time = time.time()
        logging.info(f"Processing file {file_idx}/{len(file_paths)}: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = [line.strip() for line in file if line.strip()]
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            continue
        
        if not lines:
            logging.warning(f"File {file_path} is empty or contains only whitespace. Skipping.")
            continue

        all_line_annotations = []
        logging.info(f"Processing {len(lines)} lines in batches of {BATCH_SIZE}...")
        
        for i in range(0, len(lines), BATCH_SIZE):
            line_batch = lines[i:i + BATCH_SIZE]
            
            try:
                doc = nlp(line_batch)
                sent_annotations = []
                for sent in doc.sentences:
                    word_annotations = []
                    for token, word in zip(sent.tokens, sent.words):
                        wa = {
                            'id': word.id, 'text': word.text, 'lemma': word.lemma,
                            'upos': word.upos, 'xpos': word.xpos, 'feats': word.feats,
                            'start_char': token.start_char, 'end_char': token.end_char
                        }
                        word_annotations.append(wa)

                    if args.parse:
                        constituency_parse = "(ROOT " + str(sent.constituency) + ")" if sent.constituency else None
                        sa = {'sent_text': sent.text, 'constituency_parse': constituency_parse, 'word_annotations': word_annotations}
                    else:
                        sa = {'sent_text': sent.text, 'word_annotations': word_annotations}
                    sent_annotations.append(sa)

                all_line_annotations.append({'sent_annotations': sent_annotations})
            
            except Exception as e:
                logging.error(f"Error processing batch starting at line {i} in {file_path}: {e}")
                clear_gpu_cache()
                continue

        ext = '_parsed.json' if args.parse else '.json'
        json_filename = os.path.splitext(file_path)[0] + ext
        try:
            with open(json_filename, "w", encoding='utf-8') as outfile:
                json.dump(all_line_annotations, outfile, indent=2)
            
            processing_time = time.time() - start_time
            logging.info(f"Successfully processed {file_path} in {processing_time:.2f} seconds. Output: {json_filename}")
        
        except Exception as e:
            logging.error(f"Error writing output file {json_filename}: {e}")
            continue
        
        clear_gpu_cache()

    total_time = time.time() - total_start_time
    logging.info("--- Processing Complete ---")
    if file_paths:
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        logging.info(f"Average time per file: {total_time/len(file_paths):.2f} seconds")
    log_gpu_memory()
    logging.info("All tasks finished.")