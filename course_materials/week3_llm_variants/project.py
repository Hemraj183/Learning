import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Load Pre-trained Model (GPT-2)
# ==========================================
def load_gpt2():
    print("Loading GPT-2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    return model, tokenizer

# ==========================================
# 2. Text Generation Loop (Greedy & Top-K)
# ==========================================
def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # We will generate tokens one by one
    generated = input_ids
    
    print(f"\nGenerating from prompt: '{prompt}'")
    print("-" * 50)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(generated)
            logits = outputs.logits[:, -1, :] / temperature
            
            # Top-K Sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat((generated, next_token), dim=1)
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text

# ==========================================
# 3. Inspecting the "Latent Space"
# ==========================================
def inspect_embeddings(model, tokenizer, words):
    """
    Visualize similarity between word embeddings in the model's input layer.
    """
    ids = [tokenizer.encode(w)[0] for w in words]
    ids_tensor = torch.tensor(ids)
    
    with torch.no_grad():
        # Get the embedding matrix (wte = word token embeddings)
        embeddings = model.transformer.wte(ids_tensor)
    
    # Compute Cosine Similarity Matrix
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity = torch.matmul(norm_embeddings, norm_embeddings.T)
    
    print("\nCosine Similarity Matrix:")
    print(f"{'':>10} " + " ".join([f"{w:>10}" for w in words]))
    for i, w in enumerate(words):
        row = " ".join([f"{s:.4f}" for s in similarity[i]])
        print(f"{w:>10} {row}")
        
    return similarity

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    model, tokenizer = load_gpt2()
    
    # 1. Generation Demo
    prompt = "The future of AI is"
    text = generate_text(model, tokenizer, prompt)
    print(f"\nGenerated:\n{text}")
    
    # 2. Latent Space Demo
    words = ["king", "queen", "man", "woman", "apple", "computer"]
    inspect_embeddings(model, tokenizer, words)
