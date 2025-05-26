# tag.py
# Author: Julie Kallini

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
    original_file = open(original_file)
    original_data = "".join(original_file.readlines())
    original_data = "".join(original_data.split())

    json_file = open(json_file)
    json_lines = json.load(json_file)
    json_data = ""
    for line in json_lines:
        for sent in line["sent_annotations"]:
            json_data += sent["sent_text"]
    json_data = "".join(json_data.split())

    # Test equivalence
    assert (original_data == json_data)


def __get_constituency_parse(sent, nlp):

    # Try parsing the doc
    try:
        parse_doc = nlp(sent.text)
    except:
        return None
    
    # Get set of constituency parse trees
    parse_trees = [str(sent.constituency) for sent in parse_doc.sentences]

    # Join parse trees and add ROOT
    constituency_parse = "(ROOT " + " ".join(parse_trees) + ")"
    return constituency_parse

def process_text_in_smaller_chunks(text, nlp, max_length=350):
    """Process text in smaller chunks to avoid BERT sequence length issues."""
    # Split the text into words
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    # Create chunks of appropriate size
    for word in words:
        # Add word length plus space
        word_length = len(word) + 1
        if current_length + word_length > max_length and current_chunk:
            # Start a new chunk if adding this word would exceed max_length
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # Process each chunk
    results = []
    for chunk in chunks:
        try:
            doc = nlp(chunk)
            results.append(doc)
        except Exception as e:
            logging.error(f"Error processing chunk: {e}")
            # Try with an even smaller chunk if there's an error
            if len(chunk) > 100:
                subchunks = process_text_in_smaller_chunks(chunk, nlp, max_length=max_length//2)
                results.extend(subchunks)

    return results

def safe_stanza_process(nlp, text, max_retries=3):
    """Safely process text with Stanza, handling cuDNN errors."""
    for attempt in range(max_retries):
        try:
            # Clean the text to ensure it's properly formatted
            cleaned_text = ' '.join(text.split())
            
            # Process with stanza
            doc = nlp(cleaned_text)
            return doc
            
        except RuntimeError as e:
            if "cuDNN error" in str(e) or "CUDNN_STATUS_NOT_SUPPORTED" in str(e):
                logging.warning(f"cuDNN error on attempt {attempt + 1}, trying smaller chunks...")
                
                # Try processing in smaller chunks
                try:
                    words = text.split()
                    if len(words) <= 10:  # If already very small, can't chunk further
                        raise e
                    
                    # Split into smaller chunks
                    chunk_size = max(10, len(words) // 4)
                    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
                    
                    # Process each chunk separately and combine results
                    all_sentences = []
                    for chunk in chunks:
                        chunk_text = ' '.join(chunk)
                        try:
                            chunk_doc = nlp(chunk_text)
                            all_sentences.extend(chunk_doc.sentences)
                        except RuntimeError as chunk_e:
                            if "cuDNN error" in str(chunk_e):
                                logging.warning(f"Skipping problematic chunk: {chunk_text[:50]}...")
                                continue
                            else:
                                raise chunk_e
                    
                    # Create a mock document object with the sentences
                    class MockDoc:
                        def __init__(self, sentences):
                            self.sentences = sentences
                    
                    return MockDoc(all_sentences)
                    
                except Exception as chunk_error:
                    if attempt == max_retries - 1:
                        logging.error(f"Failed to process text after {max_retries} attempts: {str(e)}")
                        raise e
                    continue
            else:
                raise e
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to process text: {str(e)}")
                raise e
            logging.warning(f"Attempt {attempt + 1} failed, retrying...")
            continue
    
    raise RuntimeError(f"Failed to process text after {max_retries} attempts")

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
    
    # Validate subset names
    invalid_subsets = [s for s in subsets if s not in valid_subsets]
    if invalid_subsets:
        logging.warning(f"Invalid subset names: {invalid_subsets}")
        logging.info(f"Valid subsets are: {valid_subsets}")
    
    # Filter files based on subsets
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
    parser.add_argument('path', type=str,
                        nargs='+', help="Path to file(s) (can include glob patterns)")
    parser.add_argument('-p', '--parse', action='store_true',
                        help="Include constituency parse")
    parser.add_argument('--max-seq-length', type=int, default=350,
                        help="Maximum sequence length for processing (default: 350)")
    parser.add_argument('--cpu-only', action='store_true',
                        help="Force CPU-only processing to avoid cuDNN issues")
    parser.add_argument('--force', action='store_true',
                        help="Force reprocessing even if output files already exist")
    parser.add_argument('--subset', type=str, nargs='+',
                        help="Process only specific dataset subsets (e.g., babylm_10M, babylm_100M, babylm_dev, babylm_test)")
    parser.add_argument('--list-subsets', action='store_true',
                        help="List available dataset subsets and exit")

    # Get args
    args = parser.parse_args()
    
    logging.info(f'use parsing: {args.parse}')

    # If list-subsets is requested, show available subsets and exit
    if args.list_subsets:
        # Expand glob patterns to get file list for subset listing
        file_paths = []
        for path_pattern in args.path:
            matched_files = glob.glob(path_pattern)
            file_paths.extend(matched_files)
        
        if file_paths:
            list_available_subsets(file_paths)
        else:
            logging.error("No files found to analyze subsets")
        sys.exit(0)

    # Set environment variables to potentially help with cuDNN issues
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Try to fix cuDNN issues
    try:
        if torch.cuda.is_available() and not args.cpu_only:
            # Set memory fraction to avoid memory issues
            torch.cuda.set_per_process_memory_fraction(0.8)
            # Empty cache
            torch.cuda.empty_cache()
            use_gpu = True
            logging.info("Using GPU processing")
        else:
            use_gpu = False
            logging.info("Using CPU-only processing...")
    except ImportError:
        use_gpu = False
        logging.info("PyTorch not available, using CPU-only processing")

    # Log initial GPU memory
    log_gpu_memory()

    try:
        # Init Stanza NLP tools with error handling
        logging.info("Initializing Stanza NLP pipeline...")
        
        # Create a processor config with controlled batch sizes
        processor_config = {
            'tokenize': {
                'batch_size': 500  # Reduced batch size
            },
            'pos': {
                'batch_size': 500,  # Reduced batch size
                'bert_finetune': False  # Disable bert fine-tuning
            },
            'lemma': {
                'batch_size': 500  # Reduced batch size
            }
        }

        # Initialize the pipeline with the config
        nlp1 = stanza.Pipeline(
            lang='id',
            processors='tokenize,pos,lemma',
            package="default",  # Use default instead of default_accurate
            processor_config=processor_config,
            use_gpu=use_gpu)

        # If constituency parse is needed, init second Stanza parser
        if args.parse:
            logging.info("Initializing constituency parsing pipeline...")
            constituency_config = {
                'tokenize': {
                    'batch_size': 500  # Reduced batch size
                },
                'pos': {
                    'batch_size': 500,  # Reduced batch size
                    'bert_finetune': False  # Disable bert fine-tuning
                },
                'constituency': {
                    'batch_size': 500  # Reduced batch size
                }
            }
            
            nlp2 = stanza.Pipeline(
                lang='id',
                processors='tokenize,pos,constituency',
                package="default",  # Use default instead of default_accurate
                processor_config=constituency_config,
                use_gpu=use_gpu)
            
    except Exception as e:
        logging.error(f"Error initializing Stanza pipeline: {e}")
        logging.info("Falling back to CPU-only minimal configuration...")
        processor_config = {
            'tokenize': {
                'batch_size': 100
            },
            'pos': {
                'batch_size': 100,
                'bert_finetune': False  # Disable bert fine-tuning
            },
            'lemma': {
                'batch_size': 100
            }
        }
        
        nlp1 = stanza.Pipeline(
            lang='id',
            processors='tokenize,pos,lemma',
            package="default",  # Use default instead of default_accurate
            processor_config=processor_config,
            use_gpu=False)  # Force CPU
        
        if args.parse:
            constituency_config = {
                'tokenize': {
                    'batch_size': 100
                },
                'pos': {
                    'batch_size': 100,
                    'bert_finetune': False
                },
                'constituency': {
                    'batch_size': 100
                }
            }
            
            nlp2 = stanza.Pipeline(
                lang='id',
                processors='tokenize,pos,constituency',
                package="default",
                processor_config=constituency_config,
                use_gpu=False)  # Force CPU

    BATCH_SIZE = 5000  # Reduced batch size
    
    # Expand glob patterns and get file list
    file_paths = []
    for path_pattern in args.path:
        matched_files = glob.glob(path_pattern)
        if not matched_files:
            logging.warning(f"No files found matching pattern '{path_pattern}'")
        file_paths.extend(matched_files)
    
    if not file_paths:
        logging.error("No files found to process")
        sys.exit(1)
    
    logging.info(f"Found {len(file_paths)} total files")
    
    # Filter files by subset if specified
    if args.subset:
        logging.info(f"Filtering files for subsets: {args.subset}")
        original_count = len(file_paths)
        file_paths = filter_files_by_subset(file_paths, args.subset)
        filtered_count = len(file_paths)
        logging.info(f"Filtered from {original_count} to {filtered_count} files")
        
        if not file_paths:
            logging.error(f"No files found for specified subsets: {args.subset}")
            logging.info("Use --list-subsets to see available subsets")
            sys.exit(1)
    
    logging.info(f"Processing {len(file_paths)} files")
    
    # Filter out already processed files if not forcing
    if not args.force:
        original_count = len(file_paths)
        file_paths = [fp for fp in file_paths if not check_output_exists(fp, args.parse)]
        skipped_count = original_count - len(file_paths)
        if skipped_count > 0:
            logging.info(f"Skipped {skipped_count} already processed files (use --force to reprocess)")
    
    if not file_paths:
        logging.info("All files already processed. Use --force to reprocess.")
        sys.exit(0)
        
    # Iterate over BabyLM files
    for file_idx, file_path in enumerate(file_paths, 1):
        start_time = time.time()
        logging.info(f"Processing file {file_idx}/{len(file_paths)}: {file_path}")
        log_gpu_memory()
        
        # Check if output JSON file already exists
        ext = '_parsed.json' if args.parse else '.json'
        json_filename = os.path.splitext(file_path)[0] + ext
        
        if os.path.exists(json_filename) and not args.force:
            logging.info(f"Output file {json_filename} already exists, skipping...")
            continue
        
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            continue

        # Strip lines and join text
        logging.info("Concatenating lines...")
        lines = [l.strip() for l in lines]
        line_batches = [lines[i:i + BATCH_SIZE]
                        for i in range(0, len(lines), BATCH_SIZE)]
        text_batches = [" ".join(l) for l in line_batches]

        logging.info(f"Created {len(text_batches)} text batches for processing")
        log_gpu_memory()

        # Iterate over lines in file and track annotations
        line_annotations = []
        logging.info("Segmenting and parsing text batches...")
        
        for batch_idx, text in enumerate(text_batches, 1):
            try:
                logging.info(f"Processing batch {batch_idx}/{len(text_batches)}")
                log_gpu_memory()
                
                # Use safe processing function
                doc = safe_stanza_process(nlp1, text)

                # Iterate over sents in the line and track annotations
                sent_annotations = []
                for sent in doc.sentences:

                    # Iterate over words in sent and track annotations
                    word_annotations = []
                    for token, word in zip(sent.tokens, sent.words):
                        wa = {
                            'id': word.id,
                            'text': word.text,
                            'lemma': word.lemma,
                            'upos': word.upos,
                            'xpos': word.xpos,
                            'feats': word.feats,
                            'start_char': token.start_char,
                            'end_char': token.end_char
                        }
                        word_annotations.append(wa)  # Track word annotation

                    # Get constituency parse if needed
                    if args.parse:
                        constituency_parse = __get_constituency_parse(sent, nlp2)
                        sa = {
                            'sent_text': sent.text,
                            'constituency_parse': constituency_parse,
                            'word_annotations': word_annotations,
                        }
                    else:
                        sa = {
                            'sent_text': sent.text,
                            'word_annotations': word_annotations,
                        }
                    sent_annotations.append(sa)  # Track sent annotation

                la = {
                    'sent_annotations': sent_annotations
                }
                line_annotations.append(la)  # Track line annotation
                
                # Clear GPU cache periodically
                if batch_idx % 10 == 0:
                    clear_gpu_cache()
                
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                logging.info("Skipping this batch and continuing...")
                continue

        # Write annotations to file as a JSON
        logging.info("Writing JSON output file...")
        ext = '_parsed.json' if args.parse else '.json'
        json_filename = os.path.splitext(file_path)[0] + ext
        try:
            with open(json_filename, "w") as outfile:
                json.dump(line_annotations, outfile, indent=4)
            
            processing_time = time.time() - start_time
            logging.info(f"Successfully processed {file_path} in {processing_time:.2f} seconds")
            logging.info(f"Output written to {json_filename}")
            
        except Exception as e:
            logging.error(f"Error writing output file {json_filename}: {e}")
            continue
        
        # Clear GPU cache after each file
        clear_gpu_cache()
        log_gpu_memory()
    
    # Final cleanup
    logging.info("Processing complete. Performing final cleanup...")
    clear_gpu_cache()
    log_gpu_memory()
    logging.info("All files processed successfully!")