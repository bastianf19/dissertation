import os
import glob
import argparse
import logging
from helper import clean_text, read_data, write_data, setup_logging

def clean_file(input_file, output_file):
    """
    Read a file, clean each line, and write the cleaned content to the output file
    """
    # Read the file
    lines = read_data(input_file)
    if not lines:
        logging.error(f"Could not read {input_file}")
        return False
    
    # Clean each line
    cleaned_lines = [clean_text(line)+'\n' for line in lines]
    
    # Write cleaned data
    try:
        write_data(cleaned_lines, output_file)
        logging.info(f"Successfully cleaned and saved: {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error writing to {output_file}: {str(e)}")
        return False

def process_directory(src_dir, target_dir):
    """
    Process all files in a directory and its subdirectories
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Track stats
    processed_count = 0
    error_count = 0
    
    # Walk through directory tree
    for root, dirs, files in os.walk(src_dir):
        # Calculate the relative path to preserve directory structure
        rel_path = os.path.relpath(root, src_dir)
        if rel_path == '.':
            rel_path = ''
        
        # Create corresponding target subdirectory
        current_target_dir = os.path.join(target_dir, rel_path)
        os.makedirs(current_target_dir, exist_ok=True)
        
        # Process each file in current directory
        for file in files:
            input_file = os.path.join(root, file)
            output_file = os.path.join(current_target_dir, file)
            
            logging.info(f"Processing {input_file}")
            
            if clean_file(input_file, output_file):
                processed_count += 1
            else:
                error_count += 1
    
    return processed_count, error_count

def main():
    parser = argparse.ArgumentParser(description='Preprocess Translated Data')
    parser.add_argument('--source_dir', type=str, required=True, help='Root directory containing translated babylm_data')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to save cleaned files')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save log files')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir=args.log_dir, logname='data_cleaning')
    
    logging.info(f"Starting data cleaning process")
    logging.info(f"Source directory: {args.source_dir}")
    logging.info(f"Target directory: {args.target_dir}")
    
    # Check if source directory exists
    if not os.path.isdir(args.source_dir):
        logging.error(f"Source directory {args.source_dir} does not exist!")
        return
    
    # Process all files in the source directory
    processed, errors = process_directory(args.source_dir, args.target_dir)
    
    logging.info(f"Data cleaning completed.")
    logging.info(f"Successfully processed: {processed} files")
    if errors > 0:
        logging.warning(f"Errors encountered: {errors} files")

if __name__ == "__main__":
    main()