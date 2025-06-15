import argparse
import os
import sys
import numpy as np
from comet import download_model, load_from_checkpoint
from difflib import SequenceMatcher
from tqdm import tqdm
import glob
import logging
from helper import setup_logging

def similarity_ratio(a, b):
    """Calculate string similarity ratio using SequenceMatcher"""
    return SequenceMatcher(None, a, b).ratio()

def evaluate_file_pair(source_file, target_file, model, similarity_threshold=0.8):
    """Evaluate a single pair of files using COMET"""
    # Read source and hypothesis files
    try:
        with open(source_file, 'r', encoding='utf-8') as src_f:
            sources = [line.strip() for line in src_f]
        
        with open(target_file, 'r', encoding='utf-8') as hyp_f:
            hypotheses = [line.strip() for line in hyp_f]
        
        # Skip empty files
        if not sources or not hypotheses:
            return None, f"Empty file: {source_file} or {target_file}"
            
        # Check if files have the same number of lines
        if len(sources) != len(hypotheses):
            return None, f"Line count mismatch: {source_file} ({len(sources)}) vs {target_file} ({len(hypotheses)})"
        
        # Prepare data for COMET
        data = [{"src": src, "mt": hyp} for src, hyp in zip(sources, hypotheses)]
        
        # Get COMET QE scores (no references needed)
        model_output = model.predict(data, batch_size=64, gpus=1)
        
        # Extract scores
        segment_scores = model_output.scores
        system_score = float(np.mean(segment_scores))
        
        # Check for potential untranslated/copied segments
        similarity_scores = [similarity_ratio(src, hyp) for src, hyp in zip(sources, hypotheses)]
        potentially_untranslated = [i for i, sim in enumerate(similarity_scores) if sim > similarity_threshold]
        
        info = {
            "total_segments": len(segment_scores),
            "potentially_untranslated": len(potentially_untranslated),
            "potential_untranslated_percentage": len(potentially_untranslated) / len(segment_scores) * 100 if segment_scores else 0
        }
        
        return system_score, info
    except Exception as e:
        return None, f"Error processing {source_file} and {target_file}: {str(e)}"

def find_matching_files(base_dir, source_pattern, target_pattern):
    """Find matching source and target files based on patterns"""
    source_files = glob.glob(os.path.join(base_dir, source_pattern))
    target_files = glob.glob(os.path.join(base_dir, target_pattern))
    
    # Extract identifiers from filenames to match them
    source_dict = {}
    for sf in source_files:
        if '.json' not in sf or '_parsed.json' not in sf:
            basename = os.path.basename(sf)
            parts = basename.split('.')
            if len(parts) > 1:
                # Use the corpus name and file type as key
                key = parts[0]  # Assumes format like 'aochildes.train'
                source_dict[key] = sf
    
    # Find matching target files
    matched_pairs = []
    for tf in target_files:
        if '.json' not in tf or '_parsed.json' not in tf:
            basename = os.path.basename(tf)
            parts = basename.split('.')
            if len(parts) > 1:
                key = parts[0]
                if key in source_dict:
                    matched_pairs.append((source_dict[key], tf, key))
    
    return matched_pairs

def main():
    parser = argparse.ArgumentParser(description="Evaluate COMET scores for babylm data")
    parser.add_argument("--base-dir", required=True, help="Base directory containing both source and translated files")
    parser.add_argument("--output", default="comet_scores.txt", help="Output file for scores")
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save log files')
    parser.add_argument("--clean", default=0, help="which file type be used; original or clean")
    parser.add_argument("--similarity-threshold", type=float, default=0.8, 
                        help="Threshold for detecting potentially untranslated text (0.0-1.0)")
    parser.add_argument("--detailed-output", default="comet_details.txt", help="Detailed output file with additional info")
    args = parser.parse_args()
    
    setup_logging(log_dir=args.log_dir, logname='comet_eval')

    # Load COMET model (do this only once)
    logging.info("loading COMET model...")
    # model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model_path = 'models/wmt22-cometkiwi-da/checkpoints/model.ckpt'
    # logging.info('cwd: ', os.getcwd())
    # print(os.getcwd())
    model = load_from_checkpoint(model_path)
    logging.info("Model loaded successfully.")
    
    logging.info(f"Clean: {args.clean}")
    
    # Define patterns for source and target files based on your folder structure
    if args.clean == 1:
        patterns = [
            # For babylm_10M
            {
                "source": "babylm_data/babylm_10M/*",
                "target": "clean_data/nllb/babylm_data_ind/babylm_10M/*",
                "prefix": "nllb_babylm_10M"
            },
            {
                "source": "babylm_data/babylm_10M/*",
                "target": "clean_data/opus-mt/babylm_data_id/babylm_10M/*",
                "prefix": "opus_babylm_10M"
            },
            # For babylm_100M
            {
                "source": "babylm_data/babylm_100M/*",
                "target": "clean_data/nllb/babylm_data_ind/babylm_100M/*",
                "prefix": "nllb_babylm_100M"
            },
            {
                "source": "babylm_data/babylm_100M/*",
                "target": "clean_data/opus-mt/babylm_data_id/babylm_100M/*",
                "prefix": "opus_babylm_100M"
            },
            # For babylm_dev
            {
                "source": "babylm_data/babylm_dev/*",
                "target": "clean_data/nllb/babylm_data_ind/babylm_dev/*",
                "prefix": "nllb_babylm_dev"
            },
            {
                "source": "babylm_data/babylm_dev/*",
                "target": "clean_data/opus-mt/babylm_data_id/babylm_dev/*",
                "prefix": "opus_babylm_dev"
            },
            # For babylm_test
            {
                "source": "babylm_data/babylm_test/*",
                "target": "clean_data/nllb/babylm_data_ind/babylm_test/*",
                "prefix": "nllb_babylm_test"
            },
            {
                "source": "babylm_data/babylm_test/*",
                "target": "clean_data/opus-mt/babylm_data_id/babylm_test/*",
                "prefix": "opus_babylm_test"
            }
        ]
    else:
        patterns = [
            # For babylm_10M
            {
                "source": "babylm_data/babylm_10M/*",
                "target": "translated/nllb/babylm_data_ind/babylm_10M/*",
                "prefix": "nllb_babylm_10M"
            },
            {
                "source": "babylm_data/babylm_10M/*",
                "target": "translated/opus-mt/babylm_data_id/babylm_10M/*",
                "prefix": "opus_babylm_10M"
            },
            # For babylm_100M
            {
                "source": "babylm_data/babylm_100M/*",
                "target": "translated/nllb/babylm_data_ind/babylm_100M/*",
                "prefix": "nllb_babylm_100M"
            },
            {
                "source": "babylm_data/babylm_100M/*",
                "target": "translated/opus-mt/babylm_data_id/babylm_100M/*",
                "prefix": "opus_babylm_100M"
            },
            # For babylm_dev
            {
                "source": "babylm_data/babylm_dev/*",
                "target": "translated/nllb/babylm_data_ind/babylm_dev/*",
                "prefix": "nllb_babylm_dev"
            },
            {
                "source": "babylm_data/babylm_dev/*",
                "target": "translated/opus-mt/babylm_data_id/babylm_dev/*",
                "prefix": "opus_babylm_dev"
            },
            # For babylm_test
            {
                "source": "babylm_data/babylm_test/*",
                "target": "translated/nllb/babylm_data_ind/babylm_test/*",
                "prefix": "nllb_babylm_test"
            },
            {
                "source": "babylm_data/babylm_test/*",
                "target": "translated/opus-mt/babylm_data_id/babylm_test/*",
                "prefix": "opus_babylm_test"
            }
        ]
    
    # Process all file pairs and collect results
    all_results = []
    all_details = []
    
    for pattern in patterns:
        logging.info(f"Processing pattern: {pattern['prefix']}")
        source_pattern = os.path.join(args.base_dir, pattern['source'])
        target_pattern = os.path.join(args.base_dir, pattern['target'])
        
        matched_pairs = find_matching_files(args.base_dir, pattern['source'], pattern['target'])
        logging.info(f"Found {len(matched_pairs)} matching file pairs")
        
        for source_file, target_file, key in tqdm(matched_pairs, desc=f"Evaluating {pattern['prefix']}"):
            score, info = evaluate_file_pair(source_file, target_file, model, args.similarity_threshold)
            
            result_key = f"{pattern['prefix']}_{key}"
            if score is not None:
                all_results.append((result_key, score))
                all_details.append((result_key, score, info))
            else:
                all_results.append((result_key, "ERROR"))
                all_details.append((result_key, "ERROR", info))
    
    # Write results to output file
    with open(args.output, 'w', encoding='utf-8') as f:
        for key, score in all_results:
            if score == "ERROR":
                f.write(f"{key} : ERROR\n")
            else:
                f.write(f"{key} : {score:.4f}\n")
    
    # Write detailed results
    with open(args.detailed_output, 'w', encoding='utf-8') as f:
        f.write("File,COMET Score,Total Segments,Potentially Untranslated,Untranslated %,Notes\n")
        for key, score, info in all_details:
            if score == "ERROR":
                f.write(f"{key},ERROR,,,{info}\n")
            else:
                f.write(f"{key},{score:.4f},{info['total_segments']},{info['potentially_untranslated']},{info['potential_untranslated_percentage']:.2f}%,\n")
    
    logging.info(f"Evaluation complete. Results written to {args.output} and {args.detailed_output}")

if __name__ == "__main__":
    main()