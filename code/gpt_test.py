import torch
import argparse
from transformers import GPT2LMHeadModel, AutoTokenizer
from utils import gpt2_original_tokenizer, gpt2_rev_tokenizer

def load_gpt2_model(model_path="gpt2"):
    """
    Load GPT-2 model and tokenizer
    
    Args:
        model_path: Path to your trained model or model name (e.g., "gpt2", "gpt2-medium")
    """
    print(f"Loading model from: {model_path}")
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    if 'babylm' not in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'reverse' in model_path:
        tokenizer = gpt2_rev_tokenizer
    else:
        tokenizer = gpt2_original_tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Move model to GPU if available
    model = model.to(device)
    
    # Set padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer, device

def generate_text(model, tokenizer, prompt, device, max_length=100, temperature=0.7, top_p=0.9, num_beams=1):
    """
    Generate text using the loaded GPT-2 model
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        prompt: Input text prompt
        device: Device to run inference on
        max_length: Maximum length of generated text
        temperature: Controls randomness (lower = more deterministic)
        top_p: Nucleus sampling parameter
        num_beams: Number of beams for beam search
    """
    # Encode the prompt and move to device
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        if num_beams > 1:
            # Use beam search for more deterministic output
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=1.2
            )
        else:
            # Use sampling
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=1.2
            )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Testing Generation GPT-2 Model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the pre-trained model directory')
    args = parser.parse_args()
    print("Loading model...")
    model, tokenizer, device = load_gpt2_model(args.model_path)
    
    print("GPT-2 model loaded successfully!")
    print("Enter 'quit' to exit\n")
    
    while True:
        # Get user input
        prompt = input("Enter your prompt: ")
        
        if prompt.lower() == 'quit':
            break
        
        # Check for generation mode
        mode = input("Generation mode (beam/sample) [sample]: ").lower()
        if mode == '':
            mode = 'sample'
        
        # Generate text
        print("\nGenerating text...")
        
        if mode == 'beam':
            generated_text = generate_text(
                model, 
                tokenizer, 
                prompt,
                device,
                max_length=len(tokenizer.encode(prompt)) + 2,  # Shorter for instructions
                temperature=0.9,
                num_beams=5
            )
        else:
            generated_text = generate_text(
                model, 
                tokenizer, 
                prompt,
                device,
                max_length=len(tokenizer.encode(prompt)) + 2,  # Shorter for instructions
                temperature=0.1,  # Lower temperature for more focused output
                top_p=0.9
            )
        
        print(f"\nGenerated text:\n{generated_text}")
        print("-" * 50)

if __name__ == "__main__":
    main()