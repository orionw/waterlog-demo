import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
import numpy as np


def format_prompt(prompt):
    """Format the prompt using HuggingFace chat template."""
    chat = [
        {"role": "system", "content": "You are a helpful assistant. Please provide a direct response."},
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False) + "<|start_header_id|>assistant<|end_header_id|>"

    
class EntropyControlLogitsProcessor(LogitsProcessor):
    """Custom logits processor that implements entropy-based token selection"""
    def __init__(self, top_n=5, entropy_threshold=1.0):
        self.top_n = top_n
        self.entropy_threshold = entropy_threshold
        # Track entropies for backtracking
        self.token_entropies = []
    
    def calculate_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs))
        return 0.0 if torch.isnan(entropy) else entropy.item()
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Handle both batched and unbatched cases
        original_shape = scores.shape
        
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(0)
            
        batch_size, vocab_size = scores.shape
        
        # Calculate and store entropy for this token
        entropy = self.calculate_entropy(scores)
        self.token_entropies.append(entropy)
        
        if entropy < self.entropy_threshold:
            try:
                # Create new scores tensor initialized to -inf
                new_scores = torch.full_like(scores, float('-inf'))
                
                # Get top N indices
                n = min(self.top_n, vocab_size)
                _, top_indices = torch.topk(scores, n, dim=-1)
                
                # Create a mask for top N indices
                mask = torch.zeros_like(scores, dtype=torch.bool)
                
                for i in range(batch_size):
                    mask[i][top_indices[i]] = True
                
                # Set equal probability (logit = 0.0) for top N tokens
                new_scores[mask] = 0.0
                
                result = new_scores.squeeze(0) if len(original_shape) == 1 else new_scores
                return result
                
            except Exception as e:
                print(f"ERROR in entropy control: {str(e)}")
                return scores.squeeze(0) if len(original_shape) == 1 else scores
                
        return scores.squeeze(0) if len(original_shape) == 1 else scores

    def get_token_entropies(self):
        return self.token_entropies
        
    def reset_entropies(self):
        self.token_entropies = []


def backtracking_generation(prompt, model, tokenizer, window_size=5, entropy_per_token=1.0, 
                           max_length=100, temperature=1.0, max_attempts=3, max_temp=2.0):
    """Generate text with backtracking when entropy drops below threshold"""
    formatted_prompt = format_prompt(prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(model.device)
    
    # Keep track of backtracking stats for debugging
    backtracking_info = {
        "attempts": 0,
        "backtrack_positions": [],
        "temperatures_used": [temperature],
        "top2_forced": 0  # Track when we forced top 2 selection
    }
    
    # Initialize variables for generation
    current_temp = temperature
    current_length = input_ids.shape[1]
    target_length = min(current_length + max_length, 2048)  # Prevent exceeding model's context
    current_window_low_entropy = False  # Flag to track current window entropy status
    
    # Track the sequence being generated
    generated_sequence = input_ids.clone()
    
    # Create logits processor to track entropy
    entropy_processor = EntropyControlLogitsProcessor(
        top_n=5,  # Default for token-level sampling
        entropy_threshold=entropy_per_token
    )
    
    # Counter for current window tokens
    window_token_count = 0
    force_top2_for_window = False
    
    while current_length < target_length:
        # Determine if we need to use top-2 forced selection
        # Only use it if we've exhausted backtracking attempts AND we're still within the problematic window
        use_top2_forced = force_top2_for_window and window_token_count < window_size
        
        if use_top2_forced:
            # Create a custom top-2 logits processor for guaranteed 1 bit of entropy
            class Top2ForcedProcessor(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    # Handle both batched and unbatched cases
                    original_shape = scores.shape
                    
                    if len(scores.shape) == 1:
                        scores = scores.unsqueeze(0)
                    
                    # Get indices of top 2 logits
                    _, top_indices = torch.topk(scores, k=2, dim=-1)
                    
                    # Create new scores tensor with -inf everywhere
                    new_scores = torch.full_like(scores, float('-inf'))
                    
                    # Set equal probability (0.0 logit) for the top 2 tokens
                    for i in range(scores.shape[0]):
                        new_scores[i, top_indices[i]] = 0.0
                    
                    # Return in the original shape
                    return new_scores.squeeze(0) if len(original_shape) == 1 else new_scores
            
            # Generate with forced top-2 selection (guarantees 1 bit of entropy)
            outputs = model.generate(
                generated_sequence,
                max_new_tokens=1,
                do_sample=True,
                temperature=1.0,  # Temperature doesn't matter here as we force equal probability
                logits_processor=LogitsProcessorList([Top2ForcedProcessor()]),
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            backtracking_info["top2_forced"] += 1
            # Increment window token counter when using top-2 forced mode
            window_token_count += 1
            
        else:
            # Normal generation with current temperature
            outputs = model.generate(
                generated_sequence,
                max_new_tokens=1,
                do_sample=True,
                temperature=current_temp,
                logits_processor=LogitsProcessorList([entropy_processor]),
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Append new token to sequence
        generated_sequence = outputs
        current_length = generated_sequence.shape[1]
        
        # Always check window entropy regardless of generation mode
        token_entropies = entropy_processor.get_token_entropies()
        
        # Check if we have enough tokens to calculate window entropy
        if len(token_entropies) >= window_size and not force_top2_for_window:
            # Calculate average entropy over the window
            window_entropies = token_entropies[-window_size:]
            avg_window_entropy = sum(window_entropies) / window_size
            window_threshold = entropy_per_token * window_size
            
            # Update our current entropy status
            current_window_low_entropy = (avg_window_entropy < window_threshold)
            
            # If entropy is low and we're not already in top-2 mode
            if current_window_low_entropy:
                # If we haven't exhausted backtracking attempts, try backtracking
                if backtracking_info["attempts"] < max_attempts:
                    # Increase temperature for next attempt (capped at max_temp)
                    current_temp = min(current_temp * 1.5, max_temp)
                    backtracking_info["attempts"] += 1
                    backtracking_info["backtrack_positions"].append(current_length)
                    backtracking_info["temperatures_used"].append(current_temp)
                    
                    # Backtrack by removing the last window_size tokens
                    generated_sequence = generated_sequence[:, :-window_size]
                    current_length = generated_sequence.shape[1]
                    
                    # Reset entropy tracking for the new attempt
                    entropy_processor.reset_entropies()
                    
                    print(f"Backtracking at position {current_length}. New temp: {current_temp:.2f}")
                else:
                    # We've exhausted backtracking attempts, switch to top-2 mode for this window only
                    print(f"Switching to top-2 forced mode for this window after exhausting backtracking attempts")
                    force_top2_for_window = True
                    window_token_count = 0  # Start counting tokens for this window
                    # Reset attempt counter for future windows
                    backtracking_info["attempts"] = 0
                    # Reset entropy processor to start fresh for this window
                    entropy_processor.reset_entropies()
        
        # If we've completed a full window with top-2 forced mode, go back to normal
        if force_top2_for_window and window_token_count >= window_size:
            print(f"Completed forced top-2 window, returning to normal generation")
            force_top2_for_window = False
            # Reset to original temperature
            current_temp = temperature
            # Reset entropy processor to start fresh
            entropy_processor.reset_entropies()
            
        # Check for EOS token
        if generated_sequence[0, -1].item() == tokenizer.eos_token_id:
            break
    
    # Decode the final generated text
    full_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=False)
    
    # Return generated text and backtracking info
    return full_text, backtracking_info


def generate_text(prompt, model, tokenizer, top_n=5, max_length=100, temperature=1.0, entropy_threshold=1.0):
    """Generate text using entropy-controlled sampling via HF generation pipeline"""
    formatted_prompt = format_prompt(prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)
    
    # Create logits processor
    entropy_processor = EntropyControlLogitsProcessor(
        top_n=top_n,
        entropy_threshold=entropy_threshold
    )
    
    # Generate using HF's built-in generation with our custom processor
    outputs = model.generate(
        inputs.input_ids.to(model.device),
        attention_mask=inputs.attention_mask.to(model.device),
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        logits_processor=LogitsProcessorList([entropy_processor]),
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    
    # Extract only the generated response
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)    
    return full_text


def compare_generations(prompt, max_length=100, window_size=5, entropy_per_token=1.0, temperature=0.1, top_n=5):
    """Compare standard and backtracking generation"""
    # Standard generation
    standard_output = model.generate(
        **tokenizer(format_prompt(prompt), return_tensors="pt", add_special_tokens=True).to(model.device),
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    standard_text = tokenizer.decode(standard_output[0], skip_special_tokens=False)    
    
    # Backtracking generation
    backtracking_response, backtracking_info = backtracking_generation(
        prompt,
        model,
        tokenizer,
        window_size=window_size,
        entropy_per_token=entropy_per_token,
        max_length=max_length,
        temperature=temperature
    )
    
    # Format backtracking info
    backtracking_stats = (
        f"Attempts: {backtracking_info['attempts']}\n"
        f"Backtrack positions: {backtracking_info['backtrack_positions']}\n"
        f"Temperatures used: {[round(t, 2) for t in backtracking_info['temperatures_used']]}\n"
        f"Top-2 forced selections: {backtracking_info['top2_forced']}"
    )
    
    # Extract just the assistant's response
    standard_response = standard_text.split("<|end_header_id|>")[-1].strip()
    backtracking_response = backtracking_response.split("<|end_header_id|>")[-1].strip()
    
    return standard_response, backtracking_response, backtracking_stats


# Create Gradio interface
demo = gr.Interface(
    fn=compare_generations,
    inputs=[
        gr.Textbox(label="Input Prompt", lines=3),
        gr.Slider(minimum=10, maximum=200, value=100, step=10, label="Max Length"),
        gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Window Size"),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Entropy Per Token"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.5, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top N for Token-Level Control"),
    ],
    outputs=[
        gr.Textbox(label="Standard Generation", lines=8),
        gr.Textbox(label="Backtracking Generation", lines=8),
        gr.Textbox(label="Backtracking Stats", lines=4),
    ],
    title="Llama3 3.1 8B Chat Generation Comparison with Backtracking",
    description="Compare three generation methods: standard, entropy-controlled, and backtracking. Backtracking monitors the entropy over a window and regenerates with higher temperature if entropy falls below threshold. When backtracking fails, it falls back to top-2 token selection for guaranteed 1 bit of entropy."
)

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    demo.launch(share=True)