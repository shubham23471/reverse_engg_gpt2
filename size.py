import torch

def get_model_size_in_gb(model):
    # Calculate the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    
    # Assume float32 type, where each parameter takes 4 bytes
    param_size_in_bytes = total_params * 4  # 4 bytes for float32
    
    # Convert to GB
    size_in_gb = param_size_in_bytes / (1024**3)  # Convert from bytes to GB
    return size_in_gb

# Example usage with GPT-2 or any other model
from transformers import GPT2Model
model = GPT2Model.from_pretrained('gpt2')

print(f"Model size: {get_model_size_in_gb(model):.2f} GB")
