import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# ==========================================
# 1. Router Architecture
# ==========================================
class RouterNetwork(nn.Module):
    def __init__(self, num_experts=3):
        super().__init__()
        # Use a small BERT model for embeddings
        self.bert = BertModel.from_pretrained('prajjwal1/bert-tiny') # Tiny BERT for speed
        self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        
        # Classifier Head: Routes input to an expert
        # 0: General Chat, 1: Code Assistant, 2: Math Solver
        self.classifier = nn.Linear(128, num_experts) # TinyBERT dim is 128
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding for classification
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits
    
    def route(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = self.forward(inputs.input_ids, inputs.attention_mask)
            expert_id = torch.argmax(logits, dim=1).item()
        return expert_id

# ==========================================
# 2. Mock Experts
# ==========================================
def expert_chat(text): return f"[Chat Expert]: I can help with general questions about '{text}'."
def expert_code(text): return f"[Code Expert]: import solution; print('Solving {text}')"
def expert_math(text): return f"[Math Expert]: Calculating result for '{text}'..."

experts = {
    0: expert_chat,
    1: expert_code,
    2: expert_math
}

expert_names = ["General Chat", "Code", "Math"]

# ==========================================
# 3. Training Loop (Simulated)
# ==========================================
def train_router(model):
    # Simulated training data
    data = [
        ("Hello, how are you?", 0),
        ("Write a python function", 1),
        ("Calculate the square root of 144", 2),
        ("What is the weather?", 0),
        ("Fix this bug in my code", 1),
        ("Solve 2x + 5 = 10", 2)
    ]
    
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Router...")
    model.train()
    for epoch in range(5):
        total_loss = 0
        for text, label in data:
            inputs = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            label_tensor = torch.tensor([label])
            
            optimizer.zero_grad()
            logits = model(inputs.input_ids, inputs.attention_mask)
            loss = criterion(logits, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    router = RouterNetwork()
    train_router(router)
    
    router.eval()
    print("\nTesting Router:")
    test_queries = [
        "Tell me a joke", 
        "Implement binary search", 
        "What is 5 times 5?"
    ]
    
    for query in test_queries:
        expert_id = router.route(query)
        print(f"Query: '{query}' -> Expert: {expert_names[expert_id]}")
        print(f"Response: {experts[expert_id](query)}\n")
