import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import logging
import sys
from datetime import datetime
import argparse
from pathlib import Path

MIN_TOKENS = 128

def setup_logging(output_dir):
    """Set up logging configuration."""
    log_file = output_dir / f'entropy_analysis_{datetime.now():%Y%m%d_%H%M%S}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze token entropy in first WildChat response')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Directory to save outputs (default: outputs)')
    parser.add_argument('--model', type=str, 
                      default='meta-llama/Llama-3.1-8B-Instruct',
                      help='HuggingFace model to use')
    parser.add_argument('--sample-size', type=int, default=1000,
                      help='Number of conversations to sample (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--split', type=str, default='train',
                      help='Dataset split to use (default: train)')
    return parser.parse_args()

def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer."""
    logging.info(f"Loading model: {model_name}")
    
    logging.info("Checking CUDA availability...")
    if torch.cuda.is_available():
        device = "cuda"
        logging.info(f"CUDA available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logging.warning("CUDA not available. Using CPU.")
    
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        use_chat_template=True
    )
    
    logging.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    logging.info(f"Model loaded successfully. Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    return model, tokenizer

def calculate_token_entropy(prompt, model, tokenizer, max_new_tokens=256):
    """Calculate per-token entropy for model-generated outputs given a prompt."""
    # Skip empty prompts
    if not prompt.strip():
        return None
        
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate tokens
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        
    generated_tokens = outputs.sequences[0, inputs.input_ids.size(1):]  # Skip prompt tokens
    scores = torch.stack(outputs.scores)  # Shape: [num_tokens, batch=1, vocab_size]
    
    # Calculate entropy for each generated token
    results = []
    for pos, (token_id, token_scores) in enumerate(zip(generated_tokens, scores)):
        # Calculate probability distribution
        probs = torch.nn.functional.softmax(token_scores[0], dim=-1)
        
        # Calculate entropy: -∑(p_i * log2(p_i))
        logs = torch.log2(probs)
        # make the -infs zero
        logs[logs == -float('inf')] = 0.0
        entropy = -torch.sum(probs * logs)

        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        results.append({
            'position': pos,
            'token': token,
            'entropy': float(entropy.cpu())
        })
    
    return results

def sample_wildchat_dataset(sample_size, seed, split):
    """Load and sample conversations from WildChat dataset."""
    logging.info("Loading WildChat dataset from HuggingFace")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("allenai/WildChat", split=split)
    logging.info(f"Loaded {len(dataset)} conversations from {split} split")
    
    # Sample conversations
    if sample_size < len(dataset):
        dataset = dataset.shuffle(seed=seed)
        sampled_dataset = dataset.select(range(sample_size))
        logging.info(f"Sampled {sample_size} conversations")
    else:
        sampled_dataset = dataset
        logging.warning(f"Sample size {sample_size} larger than dataset. Using all conversations.")
    
    return sampled_dataset

def process_wildchat_conversations(dataset, model, tokenizer, output_dir):
    """Process first human message from each WildChat conversation and generate responses."""
    all_results = []
    
    for conv_id, item in enumerate(tqdm(dataset, desc="Processing conversations")):
        conversation = item['conversation']
        
        # Extract only the first human message
        first_message = None
        for msg in conversation:
            if msg["role"] == "user":
                first_message = msg["content"]
                break
                
        if not first_message:
            logging.debug(f"No user message found in conversation {conv_id}")
            continue
            
        # Calculate entropy for the generated response
        results = calculate_token_entropy(first_message, model, tokenizer)
        if not results:  # Skip if generation failed
            continue
        
        # Add metadata to results
        for r in results:
            r['conversation_id'] = conv_id
            r['prompt'] = first_message
        
        all_results.extend(results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    output_file = output_dir / "wildchat_generation_entropy.csv"
    logging.info(f"Saving results to {output_file}")
    df.to_csv(output_file, index=False)
    
    # Calculate and save summary statistics
    summary_file = output_dir / "generation_analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("WildChat Generation Entropy Analysis Summary\n")
        f.write("==========================================\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write(f"Total conversations analyzed: {len(dataset)}\n")
        f.write(f"Total tokens generated: {len(df)}\n")
        f.write(f"Average entropy: {df['entropy'].mean():.2f} bits\n")
        f.write(f"Max entropy: {df['entropy'].max():.2f} bits\n")
        f.write(f"Min entropy: {df['entropy'].min():.2f} bits\n\n")
        
        # Token statistics
        f.write("Top 20 Highest Entropy Tokens:\n")
        top_20 = df.nlargest(20, 'entropy')
        f.write(top_20[['token', 'entropy', 'conversation_id', 'position']].to_string())
        
        # Position-based statistics
        f.write("\n\nAverage Entropy by Position (first 20 positions):\n")
        pos_stats = df.groupby('position')['entropy'].mean().head(20)
        f.write(pos_stats.to_string())
    
    logging.info(f"Summary statistics saved to {summary_file}")
    return df

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(output_dir)
    logging.info(f"Arguments: {args}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Sample dataset
    dataset = sample_wildchat_dataset(args.sample_size, args.seed, args.split)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Process conversations
    df = process_wildchat_conversations(dataset, model, tokenizer, output_dir)
    
    logging.info("Analysis completed successfully!")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Results directory: {output_dir}")

if __name__ == "__main__":
    main()