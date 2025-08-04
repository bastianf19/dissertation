import json
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2Config
import os
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import argparse
import datetime
from utils import gpt2_original_tokenizer, gpt2_rev_tokenizer

def load_model(model_path: str):
    """Load the pre-trained GPT-2 model and tokenizer."""
    print(f"Loading model from {model_path}")
    config = GPT2Config.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
    if 'reverse' not in model_path:
        print("using original tokenizer")
        tokenizer = gpt2_original_tokenizer
    else:
        print("using reverse tokenizer")
        tokenizer = gpt2_rev_tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("/home/s2587034/dissertation/models/indo_tokenizer")
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

def calculate_perplexity(model, tokenizer, device, text: str) -> float:
    """Calculate perplexity of a single sentence."""
    encodings = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss
    
    return torch.exp(loss).item()

def evaluate_blimp_file(model, tokenizer, device, file_path: str) -> Dict:
    """Evaluate model on a BLiMP JSONL file."""
    results = {
        'total': 0,
        'correct': 0,
        'by_category': {},
        'by_phenomenon': {},
        'examples': []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Evaluating {os.path.basename(file_path)}"):
            try:
                data = json.loads(line.strip())
                
                # Get perplexity for both correct and wrong sentences
                ppl_correct = calculate_perplexity(model, tokenizer, device, data['correct'])
                ppl_wrong = calculate_perplexity(model, tokenizer, device, data['wrong'])
                
                # Get category and phenomenon information
                category = data.get('category', 'unknown')
                phenomenon = data.get('linguistic_phenomenon', 'unknown')
                subcategory = data.get('subcategory', 'none')
                
                # Count as correct if the correct sentence has lower perplexity
                is_correct = ppl_correct < ppl_wrong
                results['total'] += 1
                if is_correct:
                    results['correct'] += 1
                
                # Track by category
                if category not in results['by_category']:
                    results['by_category'][category] = {'total': 0, 'correct': 0}
                results['by_category'][category]['total'] += 1
                if is_correct:
                    results['by_category'][category]['correct'] += 1
                
                # Track by linguistic phenomenon
                if phenomenon not in results['by_phenomenon']:
                    results['by_phenomenon'][phenomenon] = {'total': 0, 'correct': 0}
                results['by_phenomenon'][phenomenon]['total'] += 1
                if is_correct:
                    results['by_phenomenon'][phenomenon]['correct'] += 1
                
                # Save example with prediction
                results['examples'].append({
                    'id': data['id'],
                    'correct': data['correct'],
                    'wrong': data['wrong'],
                    'ppl_correct': ppl_correct,
                    'ppl_wrong': ppl_wrong,
                    'is_correct': is_correct,
                    'phenomenon': phenomenon,
                    'category': category,
                    'subcategory': subcategory
                })
                
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    # Calculate overall and category-wise accuracies
    if results['total'] > 0:
        results['accuracy'] = results['correct'] / results['total']
        
        # Calculate accuracy by category
        for category in results['by_category']:
            if results['by_category'][category]['total'] > 0:
                results['by_category'][category]['accuracy'] = (
                    results['by_category'][category]['correct'] / 
                    results['by_category'][category]['total']
                )
        
        # Calculate accuracy by phenomenon
        for phenomenon in results['by_phenomenon']:
            if results['by_phenomenon'][phenomenon]['total'] > 0:
                results['by_phenomenon'][phenomenon]['accuracy'] = (
                    results['by_phenomenon'][phenomenon]['correct'] / 
                    results['by_phenomenon'][phenomenon]['total']
                )
    else:
        results['accuracy'] = 0.0
    
    return results

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate GPT-2 model on Indonesian BLiMP benchmark')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the pre-trained model directory')
    parser.add_argument('--output_dir', type=str, 
                      default="/home/s2587034/dissertation/results/blimp_evaluation",
                      help='Directory to save evaluation results')
    parser.add_argument('--data_dir', type=str,
                      default="/home/s2587034/dissertation/BHASA/lindsea/id/syntax",
                      help='Directory containing BLiMP JSONL files')
    
    args = parser.parse_args()
    
    # Extract model name from the path
    model_name = os.path.basename(os.path.normpath(args.model_path))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, tokenizer, device = load_model(args.model_path)
    
    # List all BLiMP files to evaluate
    blimp_files = [
        "argument_structure.jsonl",
        "filler-gap_dependencies.jsonl",
        "morphology.jsonl",
        "NPIs_and_negation.jsonl"
    ]
    blimp_files = [os.path.join(args.data_dir, f) for f in blimp_files]
    
    # Evaluate on each file
    all_results = {}
    
    for file_path in blimp_files:
        if not os.path.exists(file_path):
            print(f"\nWarning: File not found: {file_path}")
            continue
            
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        print(f"\nEvaluating {file_name}...")
        
        results = evaluate_blimp_file(model, tokenizer, device, file_path)
        
        # Add model info to results
        results['model_name'] = model_name
        results['model_path'] = args.model_path
        results['evaluation_time'] = datetime.datetime.now().isoformat()
        
        all_results[file_name] = {
            'model': model_name,
            'accuracy': results['accuracy'],
            'total_examples': results['total'],
            'correct_predictions': results['correct']
        }
        
        # Save detailed results with model name in filename
        output_file = os.path.join(args.output_dir, f"{base_name}_{model_name}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"  Model: {model_name}")
        print(f"  Overall Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
        
        # Print accuracy by category
        if results['by_category']:
            print("\n  Accuracy by category:")
            for category, stats in results['by_category'].items():
                print(f"  - {category}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    # Save summary of all results with model name and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.output_dir, f"summary_{model_name}_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': model_name,
            'model_path': args.model_path,
            'evaluation_time': timestamp,
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print("\nEvaluation complete. Results saved to:")
    print(f"- Detailed results: {os.path.join(args.output_dir, f'{base_name}_{model_name}_results.json')}")
    print(f"- Summary: {summary_file}")

if __name__ == "__main__":
    main()