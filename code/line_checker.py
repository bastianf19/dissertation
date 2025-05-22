import os
import glob
import pandas as pd
from pathlib import Path

def get_file_length(file_path):
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def main():
    # Define the paths for original and translated directories
    original_base_dir = "dataset/babylm_data"
    # translated_base_dir = "dataset/translated/nllb/babylm_data_ind"
    translated_base_dir = "dataset/translated/opus-mt/babylm_data_id"
    print(f"target translated: {translated_base_dir}")
    
    # Directories to compare (based on the image)
    subdirs_to_check = [
        "babylm_10M", 
        "babylm_100M",
        "babylm_dev",
        "babylm_test"
    ]
    
    # Lists to store results
    results = []
    
    # Process each subdirectory
    for subdir in subdirs_to_check:
        original_subdir = os.path.join(original_base_dir, subdir)
        translated_subdir = os.path.join(translated_base_dir, subdir)
        
        # Check if both directories exist
        if not os.path.exists(original_subdir):
            print(f"Original directory not found: {original_subdir}")
            continue
            
        if not os.path.exists(translated_subdir):
            print(f"Translated directory not found: {translated_subdir}")
            continue
        
        # Get all .train files in the original directory
        train_files = glob.glob(os.path.join(original_subdir, "*"))
        
        # Process each train file
        for orig_file_path in train_files:
            file_name = os.path.basename(orig_file_path)
            trans_file_path = os.path.join(translated_subdir, file_name)
            
            # Check if translated file exists
            if not os.path.exists(trans_file_path):
                print(f"Warning: No corresponding translated file for {orig_file_path}")
                continue
            
            # Get line counts
            orig_length = get_file_length(orig_file_path)
            trans_length = get_file_length(trans_file_path)
            
            # Calculate difference and match status
            if orig_length is not None and trans_length is not None:
                diff = trans_length - orig_length
                match_status = "Match" if diff == 0 else "Mismatch"
            else:
                diff = None
                match_status = "Error"
            
            # Add result to our list
            results.append({
                "Original File": orig_file_path,
                "Translated File": trans_file_path,
                "Original Length": orig_length,
                "Translated Length": trans_length,
                "Difference": diff,
                "Status": match_status
            })
    
    # Create a DataFrame and display results
    if results:
        df = pd.DataFrame(results)
        print("\nSummary Statistics:")
        print(f"Total files checked: {len(df)}")
        print(f"Matching files: {sum(df['Status'] == 'Match')}")
        print(f"Mismatching files: {sum(df['Status'] == 'Mismatch')}")
        print(f"Files with errors: {sum(df['Status'] == 'Error')}")
        
        # Save results to CSV
        df.to_csv("translation_length_comparison.csv", index=False)
        print("\nDetailed results saved to translation_length_comparison.csv")
        
        # Display files with mismatches
        if sum(df['Status'] == 'Mismatch') > 0:
            print("\nFiles with length mismatches:")
            mismatches = df[df['Status'] == 'Mismatch']
            for _, row in mismatches.iterrows():
                print(f"  {Path(row['Original File']).name}: Original={row['Original Length']}, "
                      f"Translated={row['Translated Length']}, Diff={row['Difference']}")
    else:
        print("No files were compared. Check if directories exist and contain .train files.")

if __name__ == "__main__":
    main()