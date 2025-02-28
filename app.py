import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList



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
    
    def calculate_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs))
        return 0.0 if torch.isnan(entropy) else entropy.item()
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # print("\n=== Debug EntropyControlLogitsProcessor ===")
        # print(f"Input scores shape: {scores.shape}")
        # print(f"Input scores device: {scores.device}")
        # print(f"First few logits: {scores[:5]}")
        
        # Handle both batched and unbatched cases
        original_shape = scores.shape
        # print(f"Original shape: {original_shape}")
        
        if len(scores.shape) == 1:
            # print("Unbatched input - adding batch dimension")
            scores = scores.unsqueeze(0)
            
        batch_size, vocab_size = scores.shape
        # print(f"Working shape - batch_size: {batch_size}, vocab_size: {vocab_size}")
        
        entropy = self.calculate_entropy(scores)
        # print(f"Calculated entropy: {entropy}")
        
        if entropy < self.entropy_threshold:
            # print("Entropy below threshold - applying top-N filtering")
            try:
                # Create new scores tensor initialized to -inf
                new_scores = torch.full_like(scores, float('-inf'))
                # print(f"Created new scores tensor with shape: {new_scores.shape}")
                
                # Get top N indices
                n = min(self.top_n, vocab_size)
                # print(f"Getting top {n} tokens")
                _, top_indices = torch.topk(scores, n, dim=-1)
                # print(f"Top indices shape: {top_indices.shape}")
                # print(f"Top indices: {top_indices}")
                
                # Create a mask for top N indices
                mask = torch.zeros_like(scores, dtype=torch.bool)
                # print(f"Created mask with shape: {mask.shape}")
                
                for i in range(batch_size):
                    mask[i][top_indices[i]] = True
                
                # print(f"Mask sum (should equal top_n): {mask.sum().item()}")
                
                # Set equal probability (logit = 0.0) for top N tokens
                new_scores[mask] = 0.0
                
                # Verify the changes
                num_selected = (new_scores > float('-inf')).sum().item()
                # print(f"Number of tokens selected: {num_selected}")
                # print(f"Should equal top_n: {self.top_n}")
                
                result = new_scores.squeeze(0) if len(original_shape) == 1 else new_scores
                # print(f"Final output shape: {result.shape}")
                return result
                
            except Exception as e:
                print(f"ERROR in entropy control: {str(e)}")
                print(f"Returning original scores as fallback")
                return scores.squeeze(0) if len(original_shape) == 1 else scores
                
        # print("Entropy above threshold - returning original scores")
        return scores.squeeze(0) if len(original_shape) == 1 else scores

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

def compare_generations(prompt, max_length=100, top_n=2, temperature=0.1):
    """Compare standard and entropy-controlled generation"""
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
    
    # Entropy-controlled generation
    entropy_response = generate_text(
        prompt,
        model,
        tokenizer,
        top_n=top_n,
        max_length=max_length,
        temperature=temperature
    )
    
    return standard_text.split("<|end_header_id|>")[-1].strip(), entropy_response.split("<|end_header_id|>")[-1].strip()

# Create Gradio interface
demo = gr.Interface(
    fn=compare_generations,
    inputs=[
        gr.Textbox(label="Input Prompt", lines=3),
        gr.Slider(minimum=10, maximum=200, value=100, step=10, label="Max Length"),
        gr.Slider(minimum=1, maximum=20, value=2, step=1, label="Top N for Entropy Control"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.1, step=0.1, label="Temperature"),
    ],
    outputs=[
        gr.Textbox(label="Standard Generation", lines=8),
        gr.Textbox(label="Entropy-Controlled Generation", lines=8),
    ],
    title="Llama3 3.1 8B Chat Generation Comparison",
    description="Compare standard generation with entropy-controlled generation. When entropy drops below 1 bit, the model samples from the top N tokens to maintain diversity."
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