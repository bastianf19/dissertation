import os
import json
import yaml
import torch
import argparse
from transformers import GPT2LMHeadModel, AutoTokenizer
from utils import PERTURBATIONS
from functools import partial
import datetime
import re
import traceback

class GPT2Generator:
    def __init__(self, model_path, model_variant='shuffle_control'):
        """
        Initialize the GPT-2 model and tokenizer from custom paths.
        
        Args:
            model_path (str): Path to the saved model directory
            model_variant (str): Type of model variant (e.g., 'shuffle_control', 'shuffle_local3', 'reverse_control')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_variant = model_variant
        
        # Initialize perturbation functions based on model variant
        self._setup_perturbation_functions()
        
        # Load tokenizer and model
        self._load_tokenizer_and_model(model_path)
    
    def _setup_perturbation_functions(self):
        """Setup perturbation functions based on model variant."""
        # Get perturbation config from PERTURBATIONS
        if self.model_variant not in PERTURBATIONS:
            raise ValueError(f"Unknown model variant: {self.model_variant}")
            
        perturbation_config = PERTURBATIONS[self.model_variant]
        self.perturbation_function = perturbation_config["perturbation_function"]
        self.affect_function = perturbation_config["affect_function"]
        self.filter_function = perturbation_config["filter_function"]
        self.tokenizer = perturbation_config["gpt2_tokenizer"]
    
    def _load_tokenizer_and_model(self, model_path):
        """Load tokenizer and model with appropriate settings based on variant."""
        try:
            # Load model
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_text(self, text):
        """
        Preprocess text according to the model variant's perturbation function.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Return early if no tokenizer or no perturbation function
        if not hasattr(self, 'tokenizer') or not hasattr(self, 'perturbation_function'):
            return text
            
        try:
            # Create a dictionary with 'sent_text' key as expected by perturbation functions
            sent_dict = {"sent_text": text}
            
            # Apply perturbation function to the sentence dictionary
            # The function might return tokens or a processed sentence
            result = self.perturbation_function(sent_dict)
            
            # Handle different return types from perturbation functions
            if isinstance(result, dict) and 'sent_text' in result:
                # If the function returns a dict with 'sent_text', use that
                return result['sent_text']
            elif isinstance(result, list):
                # If it returns a list of tokens, convert back to string
                if all(isinstance(t, int) for t in result):
                    # List of token IDs
                    return self.tokenizer.decode(result, skip_special_tokens=True)
                elif all(isinstance(t, str) for t in result):
                    # List of token strings
                    return self.tokenizer.convert_tokens_to_string(result)
            
            # If we get here, return the original text
            return text
                
        except Exception as e:
            print(f"Error in preprocess_text: {e}")
            traceback.print_exc()
            
        # Return original text if anything goes wrong
        return text
    
    def generate_text(self, prompt, max_length=100, num_return_sequences=1, 
                     temperature=0, top_p=0.9, top_k=50, do_sample=True, verbose=True):
        """
        Generate text based on input prompt.
        
        Args:
            prompt (str): Input text prompt
            max_length (int): Maximum length of generated text
            num_return_sequences (int): Number of sequences to generate
            temperature (float): Sampling temperature (higher = more random)
            top_p (float): Nucleus sampling parameter
            top_k (int): Top-k sampling parameter
            do_sample (bool): Whether to use sampling or greedy decoding
            verbose (bool): Whether to print debugging information
            
        Returns:
            list: Generated text sequences
        """
        # Preprocess prompt
        original_prompt = prompt
        prompt = self.preprocess_text(prompt)
        
        if verbose and prompt != original_prompt:
            print("\n=== Preprocessed Prompt ===")
            print("Original:", original_prompt)
            print("Preprocessed:", prompt)
            print("=" * 30 + "\n")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            generated_sequence = output[input_ids.shape[-1]:]
            text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts

    def calculate_perplexity(self, text, verbose=True):
        """
        Calculate perplexity of the given text using the model.
        Lower perplexity means the text is more likely according to the model.
        
        This implementation follows the same technique as in perplexities.py.
        
        Args:
            text (str): Input text to calculate perplexity for
            verbose (bool): Whether to print debugging information
            
        Returns:
            float: Perplexity score
        """
        # Preprocess text
        original_text = text
        text = self.preprocess_text(text)
        
        if verbose and text != original_text:
            print("\n=== Preprocessed Text for Perplexity ===")
            print("Original:", original_text)
            print("Preprocessed:", text)
            print("=" * 50 + "\n")
        
        # Tokenize the text
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.to(self.device)
        labels = input_ids.clone()
        attention_mask = encodings.attention_mask.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # The "shifted" nature of labels in GPT-2 (next token prediction)
            # Shift logits, labels, and attention mask by one position
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            
            # Calculate per-token loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
            
            # Reshape to get per-token losses
            loss = loss.view(shift_labels.size())
            
            # Apply the attention mask - only calculate loss where mask is 1
            loss = loss * shift_attention_mask
            
            # Calculate average loss over non-padded tokens
            if shift_attention_mask.sum() > 0:
                avg_loss = loss.sum() / shift_attention_mask.sum()
                perplexity = torch.exp(avg_loss).item()
            else:
                perplexity = float('inf')
            
            if verbose:
                print(f"Loss: {avg_loss.item() if 'avg_loss' in locals() else 'N/A'}, Perplexity: {perplexity:.4f}")
            
            return perplexity
    
    def calculate_continuation_probability(self, prompt, continuation, verbose=True):
        """
        Calculate how likely the continuation is to follow the prompt
        using the model's next-token prediction capabilities.
        
        Args:
            prompt (str): The initial prompt text
            continuation (str): The continuation text to evaluate
            verbose (bool): Whether to print debugging information
            
        Returns:
            float: Average log probability per token of the continuation
        """
        # Preprocess prompt and continuation
        original_prompt, original_continuation = prompt, continuation
        prompt = self.preprocess_text(prompt)
        continuation = self.preprocess_text(continuation)
        
        if verbose and (prompt != original_prompt or continuation != original_continuation):
            print("\n=== Preprocessed Inputs for Continuation Probability ===")
            print("Original Prompt:", original_prompt)
            print("Preprocessed Prompt:", prompt)
            print("\nOriginal Continuation:", original_continuation)
            print("Preprocessed Continuation:", continuation)
            print("=" * 50 + "\n")
        
        # Encode the prompt and continuation
        full_text = prompt + "\n" + continuation
        encodings = self.tokenizer(full_text, return_tensors='pt').to(self.device)
        input_ids = encodings.input_ids
        
        # Get the length of the prompt to know where the continuation starts
        prompt_encodings = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        prompt_length = prompt_encodings.input_ids.shape[1]
        
        # Get logits for all positions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            logits = outputs.logits
            
            # Calculate log probabilities for the continuation part
            # Shift input_ids and logits to align predictions with next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate loss only for the continuation part
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
            
            # Reshape to get per-token losses
            loss = loss.view(shift_labels.size(0), -1)
            
            # Take only the continuation part (after the prompt)
            if prompt_length > 0:
                loss = loss[:, prompt_length-1:-1]  # -1 because of the shift
            
            # Calculate average log probability (negative loss)
            avg_log_prob = -loss.mean().item()
            
        return avg_log_prob

    def choose_best_option(self, prompt, option_a, option_b, method='perplexity', return_perplexity=False, verbose=True):
        """
        Choose between two options based on the specified method.
        
        Args:
            prompt (str): The prompt/context
            option_a (str): First option
            option_b (str): Second option
            method (str): 'perplexity' or 'continuation_probability'
            return_perplexity (bool): Whether to return the perplexity of the chosen option
            verbose (bool): Whether to print debugging information
            
        Returns:
            str: 'A' or 'B' indicating the better option
            float: (optional) Perplexity of the chosen option
        """
        # Preprocess prompt and options
        original_prompt, original_opt_a, original_opt_b = prompt, option_a, option_b
        prompt = self.preprocess_text(prompt)
        option_a = self.preprocess_text(option_a)
        option_b = self.preprocess_text(option_b)
        
        if verbose and (prompt != original_prompt or 
                       option_a != original_opt_a or 
                       option_b != original_opt_b):
            print("\n=== Preprocessed Inputs for Option Selection ===")
            print("Original Prompt:", original_prompt)
            print("Preprocessed Prompt:", prompt)
            print("\nOriginal Option A:", original_opt_a)
            print("Preprocessed Option A:", option_a)
            print("\nOriginal Option B:", original_opt_b)
            print("Preprocessed Option B:", option_b)
            print("=" * 50 + "\n")
        
        if method == 'continuation_probability':
            prob_a = self.calculate_continuation_probability(prompt, f"A: {option_a}")
            prob_b = self.calculate_continuation_probability(prompt, f"B: {option_b}")
            print(f"\n[Continuation Prob] A: {prob_a:.4f}, B: {prob_b:.4f}")
            choice = 'A' if prob_a > prob_b else 'B'
            perplexity = -prob_a if choice == 'A' else -prob_b  # Convert to positive perplexity
        else:
            ppl_a = self.calculate_perplexity(f"{prompt}\nA: {option_a}")
            ppl_b = self.calculate_perplexity(f"{prompt}\nB: {option_b}")
            print(f"\n[Perplexity] A: {ppl_a:.4f}, B: {ppl_b:.4f}")
            choice = 'A' if ppl_a < ppl_b else 'B'
            prob_a = -ppl_a
            prob_b = -ppl_b
            perplexity = ppl_a if choice == 'A' else ppl_b
        
        # Add small noise to prevent ties
        noise = torch.randn(1).item() * 0.01
        if choice == 'A':
            prob_a += noise
        else:
            prob_b += noise
        
        if return_perplexity:
            return choice, perplexity
        return choice
    
def load_jsonl(file_path):
    """
    Load a JSONL file and return a list of dictionaries.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        list: List of dictionaries containing the JSONL data
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON on line {len(data) + 1}: {e}")
        return None

def process_pragmatic_data(pair_data, single_data, prompts_data, language='translated'):
    """
    Process both single and pair pragmatic reasoning data with prompts from YAML.
    
    Args:
        pair_data (list): List of dictionaries from the pair JSONL file
        single_data (list): List of dictionaries from the single JSONL file
        prompts_data (dict): Loaded YAML prompts data
        language (str): 'en' for English or 'translated' for Indonesian prompts
        
    Returns:
        tuple: (processed_pairs, processed_singles) - Lists of dictionaries with formatted prompts
    """

    pair_template = prompts_data.get('pragmatics', {}).get('pragmatic_reasoning_pair', {})\
        .get(language, {}).get('human', '')
    pair_system = prompts_data.get('pragmatics', {}).get('pragmatic_reasoning_pair', {})\
        .get(language, {}).get('system', '')
    
    processed_pairs = []
    for item in pair_data:
        prompt = pair_template.format(
            text=item['text'],
            conclusion=item['conclusion']
        )
        
        processed_pairs.append({
            'id': item['id'],
            'type': 'pair',
            'linguistic_phenomenon': item['linguistic_phenomenon'],
            'category': item['category'],
            'system': pair_system,
            'prompt': prompt,
            'expected_label': 'True' if item['label'] else 'False',
            'text': item['text'],
            'conclusion': item['conclusion']
        })
    
    # single_template = prompts_data.get('pragmatics', {}).get('pragmatic_reasoning_single', {})\
    #     .get(language, {}).get('human', '')
    single_template = prompts_data.get('pragmatics', {}).get('pragmatic_reasoning_single_forced', {})\
        .get(language, {}).get('human', '')

    single_system = prompts_data.get('pragmatics', {}).get('pragmatic_reasoning_single', {})\
        .get(language, {}).get('system', '')
    processed_singles = []
    for item in single_data:
        if language == 'en':
            prompt = single_template.format(
                question=item['question'],
                text=item['text'],
                choices=item['choices']
            )
        else:
            prompt = single_template.format(
                question_translated=item['question_translated'],
                text=item['text'],
                choices_translated=item['choices_translated']
            )
        
        processed_singles.append({
            'id': item['id'],
            'type': 'single',
            'linguistic_phenomenon': item['linguistic_phenomenon'],
            'category': item['category'],
            'lexical_item': item.get('lexical_item', ''),
            'system': single_system,
            'prompt': prompt,
            'expected_label': item['label'],
            'text': item['text']
        })
    
    return processed_pairs, processed_singles

def process_syntax_data(syntax_data, prompts_data, language='translated'):
    """
    Process syntax data with prompts from YAML.
    
    Args:
        syntax_data (list): List of dictionaries from the syntax JSONL files
        prompts_data (dict): Loaded YAML prompts data
        language (str): 'en' for English or 'translated' for Indonesian prompts
        
    Returns:
        list: List of dictionaries with formatted prompts and expected answers
    """
    processed_data = []
    
    for item in syntax_data:
        try:
            # Use a default task type if not specified
            task_type = item.get('task', 'minimal_pairs')
            
            # Check if we have a prompt template for this task type
            if 'syntax' not in prompts_data or task_type not in prompts_data['syntax']:
                print(f"Warning: No prompt template found for task type '{task_type}'. Using default template.")
                if 'minimal_pairs' in prompts_data.get('syntax', {}):
                    task_type = 'minimal_pairs'
                else:
                    print("Error: No valid task type found in prompts data. Skipping item.")
                    continue
            
            # Get the language-specific prompt template
            task_templates = prompts_data['syntax'].get(task_type, {})
            if language not in task_templates:
                print(f"Warning: No {language} prompt found for task type '{task_type}'. Using English prompt instead.")
                if 'en' in task_templates:
                    prompt_template = task_templates['en']
                else:
                    print(f"Error: No English prompt found for task type '{task_type}'. Skipping item.")
                    continue
            else:
                prompt_template = task_templates[language]
            
            # Get the system and human prompt templates
            system_prompt = prompt_template.get('system', '')
            human_prompt = prompt_template.get('human', '')
            
            if not human_prompt:
                print(f"Error: No human prompt template found for task type '{task_type}'. Skipping item.")
                continue
            
            # Format the prompt with the item's data
            try:
                # Get the preprocessed human prompt
                formatted_prompt = human_prompt.format(**item)
                
                # Get options and correct answer, with fallbacks
                option_a = item.get('option_a', item.get('correct', ''))
                option_b = item.get('option_b', item.get('wrong', ''))
                correct_answer = item.get('correct_answer', 'A')  # Default to 'A' if not specified
                
                # Create the processed item with original and preprocessed text
                processed_item = {
                    'id': item.get('id', ''),
                    'task': task_type,
                    'system': system_prompt,
                    'prompt': formatted_prompt,
                    'option_a': option_a,
                    'option_b': option_b,
                    'correct_answer': correct_answer,
                    'preprocessed_prompt': formatted_prompt,  # Will be updated during evaluation
                    'preprocessed_option_a': option_a,  # Will be updated during evaluation
                    'preprocessed_option_b': option_b,  # Will be updated during evaluation
                    'model_choice': None,
                    'is_correct': None
                }
                
                # Add any additional fields from the original item
                for key, value in item.items():
                    if key not in processed_item and key not in ['prompt', 'option_a', 'option_b', 'correct_answer', 'correct', 'wrong']:
                        processed_item[key] = value
                
                processed_data.append(processed_item)
                
            except KeyError as e:
                print(f"Error formatting prompt for item {item.get('id', 'unknown')}: {e}")
                print(f"Missing key in item data: {e}")
                continue
                
        except Exception as e:
            print(f"Unexpected error processing item: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return processed_data

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Evaluate GPT-2 model on BHASA')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the pre-trained model directory')
    parser.add_argument('--output_dir', type=str, 
                      default="/home/s2587034/dissertation/results/syntax_evaluation",
                      help='Directory to save evaluation results')
    parser.add_argument('--eval_type', type=str,
                      default="syntax",
                      choices=["pragmatics", "syntax"],
                      help='Type of evaluation to perform')
    parser.add_argument('--eval_method', type=str,
                      default="perplexity",
                      choices=["perplexity", "continuation_probability"],
                      help='Evaluation method to use')
    parser.add_argument('--prompt_path', type=str,
                      default="/home/s2587034/dissertation/BHASA/lindsea/id/prompts.yaml",
                      help='Path to the prompts YAML file')
    parser.add_argument('--syntax_dir', type=str,
                      default="/home/s2587034/dissertation/BHASA/lindsea/id/syntax",
                      help='Directory containing syntax JSONL files')
    parser.add_argument('--language', type=str,
                      default="translated",
                      choices=["en", "translated"],
                      help='Language for prompts (en or translated)')
    parser.add_argument('--model_variant', type=str,
                      default="shuffle_control",
                      choices=list(PERTURBATIONS.keys()),
                      help='Model variant to use')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print evaluation settings
    variant_name = args.model_path.split('/')[-2] if 'checkpoint-' in args.model_path.split('/')[-1] else args.model_path.split('/')[-1]
    print(f"\n{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"Variant: {variant_name}")
    print(f"Evaluation type: {args.eval_type}")
    print(f"Evaluation method: {args.eval_method}")
    print(f"Language: {args.language}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*50}\n")

    # Load prompts
    try:
        with open(args.prompt_path, 'r', encoding='utf-8') as file:
            prompts_data = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file {args.prompt_path} was not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)
    
    if args.eval_type == "pragmatics":
        print("Evaluating on pragmatics...")
        # load both pragmatics pair and single
        pair_data = load_jsonl("/home/s2587034/dissertation/BHASA/lindsea/id/pragmatics/pragmatic_reasoning_pair.jsonl")
        single_data = load_jsonl("/home/s2587034/dissertation/BHASA/lindsea/id/pragmatics/pragmatic_reasoning_single.jsonl")
        # print(pair_data[0].keys(), single_data[0].keys())
        processed_pairs, processed_singles = process_pragmatic_data(pair_data, single_data, prompts_data, language='en')
        print("\n=== Pair Example ===")
        print(processed_pairs[0].get('system', '') + "\n" + processed_pairs[0]['prompt'])
        print("Expected label:", processed_pairs[0]['expected_label'])
        
        print("\n=== Single Example ===")
        print(processed_singles[0].get('system', '') + "\n" + processed_singles[0]['prompt'])
        print("Expected label:", processed_singles[0]['expected_label'])
        
        # Evaluation code for pragmatics...
        
    elif args.eval_type == "syntax":
        print("Evaluating on syntax...")
        
        # Process all JSONL files in the syntax directory
        results = {}
        file_metrics = {}
        
        for filename in os.listdir(args.syntax_dir):
            if not filename.endswith('.jsonl'):
                continue
                    
            file_path = os.path.join(args.syntax_dir, filename)
            print(f"\n{'='*50}")
            print(f"Processing file: {filename}")
            print(f"{'='*50}")
            
            # Load and process data for this file
            data = load_jsonl(file_path)
            processed_data = process_syntax_data(data, prompts_data, language=args.language)
            
            file_results = []
            correct_count = 0
            total_perplexity = 0.0
            
            for item in processed_data:
                system_prompt = item.get('system', '')
                user_prompt = item['prompt'].replace(system_prompt, '').strip()
                
                # Extract options
                parts = user_prompt.split("\n")
                option_a = parts[1][3:]  # Remove 'A: ' prefix
                option_b = parts[2][3:]  # Remove 'B: ' prefix
                
                clean_prompt = system_prompt + "\n" + parts[0] + "\n"
                
                # Initialize generator
                try:
                    generator = GPT2Generator(args.model_path, model_variant=args.model_variant)
                    
                    # Get model's choice and perplexity
                    model_choice, perplexity = generator.choose_best_option(
                        clean_prompt, 
                        option_a, 
                        option_b,
                        method=args.eval_method,
                        return_perplexity=True,
                        verbose=True
                    )
                    
                    is_correct = model_choice == item['correct_answer']
                    if is_correct:
                        correct_count += 1
                    
                    total_perplexity += perplexity
                    
                    # Calculate perplexity for both options for display
                    if args.eval_method == 'perplexity':
                        ppl_a = generator.calculate_perplexity(f"{clean_prompt}A: {option_a}", verbose=True)
                        ppl_b = generator.calculate_perplexity(f"{clean_prompt}B: {option_b}", verbose=True)
                    else:
                        ppl_a = -generator.calculate_continuation_probability(clean_prompt, f"A: {option_a}", verbose=True)
                        ppl_b = -generator.calculate_continuation_probability(clean_prompt, f"B: {option_b}", verbose=True)
                    
                    # Print detailed results for this question
                    print("\n" + "="*80)
                    print(f"Question {item['id']} ({item['category']} - {item['subcategory']})")
                    print("-"*80)
                    print(f"Prompt: {clean_prompt.strip()}")
                    print(f"Option A: {option_a}")
                    print(f"Option B: {option_b}")
                    print("-"*80)
                    print(f"Expected: {item['correct_answer']}")
                    print(f"Perplexity A: {ppl_a:.2f}")
                    print(f"Perplexity B: {ppl_b:.2f}")
                    print(f"Model chose: {model_choice}")
                    print(f"Correct: {is_correct}")
                    print("="*80)
                    
                    result = {
                        'id': item.get('id', ''),
                        'category': item.get('category', ''),
                        'subcategory': item.get('subcategory', ''),
                        'prompt': clean_prompt,
                        'option_a': option_a,
                        'option_b': option_b,
                        'expected': item.get('correct_answer', 'A'),
                        'model_choice': model_choice,
                        'is_correct': is_correct,
                        'perplexity': perplexity,
                        'correct_sentence': item.get('correct_sentence', item.get('correct', '')),
                        'wrong_sentence': item.get('wrong_sentence', item.get('wrong', ''))
                    }
                    file_results.append(result)
                
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate file metrics
            file_accuracy = correct_count / len(processed_data) if processed_data else 0.0
            avg_perplexity = total_perplexity / len(processed_data) if processed_data else 0.0
            
            file_metrics[filename] = {
                'num_examples': len(processed_data),
                'accuracy': file_accuracy,
                'average_perplexity': avg_perplexity
            }
            
            results[filename] = file_results
            
            print(f"\nFile: {filename}")
            print(f"Accuracy: {file_accuracy:.4f}")
            print(f"Average Perplexity: {avg_perplexity:.4f}")
        
        # Calculate overall metrics
        total_examples = sum(m['num_examples'] for m in file_metrics.values())
        overall_accuracy = sum(m['accuracy'] * m['num_examples'] for m in file_metrics.values()) / total_examples if total_examples > 0 else 0.0
        overall_perplexity = sum(m['average_perplexity'] * m['num_examples'] for m in file_metrics.values()) / total_examples if total_examples > 0 else 0.0
        
        # Prepare final results
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        output_file = os.path.join(args.output_dir, f"syntax_eval_{variant_name}_{timestamp}.json")
        
        result_data = {
            'model_path': args.model_path,
            'variant_name': variant_name,
            'eval_type': 'syntax',
            'language': args.language,
            'eval_method': args.eval_method,
            'timestamp': timestamp,
            'overall_metrics': {
                'total_examples': total_examples,
                'accuracy': overall_accuracy,
                'average_perplexity': overall_perplexity
            },
            'file_metrics': file_metrics,
            'detailed_results': results
        }
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
            
        print(f"\n{'='*50}")
        print(f"Evaluation completed!")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Overall Average Perplexity: {overall_perplexity:.4f}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*50}")
