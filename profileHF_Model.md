#### Calculate flops and memory requirements for each layer, Given sequence length and Max Tokens
```python
import torch
from transformers import LlamaForCausalLM
from typing import List, Tuple
import math

def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def estimate_flops_linear(in_features: int, out_features: int) -> int:
    return in_features * out_features

def estimate_flops_attention(hidden_size: int, num_heads: int, seq_length: int) -> int:
    head_dim = hidden_size // num_heads
    # Q, K, V projections
    flops = 3 * estimate_flops_linear(hidden_size, hidden_size)
    # Attention score calculation
    flops += 2 * num_heads * seq_length * head_dim
    # Softmax (approximate)
    flops += num_heads * seq_length * (math.log2(seq_length) + seq_length)
    # Output projection
    flops += estimate_flops_linear(hidden_size, hidden_size)
    return flops

def estimate_flops_mlp(hidden_size: int, intermediate_size: int) -> int:
    return (2 * estimate_flops_linear(hidden_size, intermediate_size) + 
            estimate_flops_linear(intermediate_size, hidden_size) + 
            2 * intermediate_size)  # SiLU activation

def analyze_layer(layer: torch.nn.Module, hidden_size: int, num_heads: int, seq_length: int) -> Tuple[int, int]:
    total_flops = 0
    total_params = count_parameters(layer)
    
    # Self-attention
    total_flops += estimate_flops_attention(hidden_size, num_heads, seq_length)
    
    # MLP
    intermediate_size = layer.mlp.gate_proj.out_features
    total_flops += estimate_flops_mlp(hidden_size, intermediate_size)
    
    # Layer norms (approximate)
    total_flops += 2 * hidden_size * 2  # Two layer norms per transformer layer
    
    return total_flops, total_params

def analyze_model(model: LlamaForCausalLM, seq_length: int, max_new_tokens: int) -> List[Tuple[str, int, int, float]]:
    results = []
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    
    # Embedding layer
    embed_params = count_parameters(model.model.embed_tokens)
    embed_flops = hidden_size * seq_length
    embed_memory = embed_params * 2 / (1024 * 1024)  # Assuming float16 for inference
    results.append(("Embedding", embed_flops, embed_params, embed_memory))
    
    # Transformer layers
    for i, layer in enumerate(model.model.layers):
        flops, params = analyze_layer(layer, hidden_size, num_heads, seq_length)
        memory = params * 2 / (1024 * 1024)  # Assuming float16 for inference
        results.append((f"Layer {i}", flops, params, memory))
    
    # Final layer norm
    final_ln_params = count_parameters(model.model.norm)
    final_ln_flops = hidden_size * 2
    final_ln_memory = final_ln_params * 2 / (1024 * 1024)  # Assuming float16 for inference
    results.append(("Final LayerNorm", final_ln_flops, final_ln_params, final_ln_memory))
    
    return results

def calculate_inference_stats(results: List[Tuple[str, int, int, float]], seq_length: int, max_new_tokens: int) -> List[Tuple[str, int, float]]:
    inference_stats = []
    
    for layer_name, flops, params, memory in results:
        if layer_name == "Embedding":
            
            inference_flops = flops
        elif layer_name == "Final LayerNorm":
            
            inference_flops = flops * max_new_tokens
        else:
            
            inference_flops = flops * (1 + max_new_tokens / seq_length)
        
        inference_stats.append((layer_name, inference_flops, memory))
    
    return inference_stats

def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  
    model = LlamaForCausalLM.from_pretrained(model_name)
    seq_length = 500
    max_new_tokens = 28
    
    results = analyze_model(model, seq_length, max_new_tokens)
    inference_stats = calculate_inference_stats(results, seq_length, max_new_tokens)
    
    total_flops = 0
    total_memory = 0
    print(f"{'Layer':<20} {'Inference FLOPs':<20} {'Memory (MB)':<15}")
    print("-" * 55)
    for layer_name, flops, memory in inference_stats:
        print(f"{layer_name:<20} {flops:<20,.0f} {memory:<15.2f}")
        total_flops += flops
        total_memory += memory
    
    print("-" * 55)
    print(f"{'Total':<20} {total_flops:<20,.0f} {total_memory:<15.2f}")
    print(f"\nTotal Inference FLOPs for {max_new_tokens} new tokens: {total_flops:,.0f}")
    print(f"Total Memory Usage: {total_memory:.2f} MB")


main()

```


#### Output:

```bash
Layer                Inference FLOPs      Memory (MB)    
-------------------------------------------------------
Embedding            2,048,000            250.00         
Layer 0              226,673,590          386.02         
Layer 1              226,673,590          386.02         
Layer 2              226,673,590          386.02         
Layer 3              226,673,590          386.02         
Layer 4              226,673,590          386.02         
Layer 5              226,673,590          386.02         
Layer 6              226,673,590          386.02         
Layer 7              226,673,590          386.02         
Layer 8              226,673,590          386.02         
Layer 9              226,673,590          386.02         
Layer 10             226,673,590          386.02         
Layer 11             226,673,590          386.02         
Layer 12             226,673,590          386.02         
Layer 13             226,673,590          386.02         
Layer 14             226,673,590          386.02         
Layer 15             226,673,590          386.02         
Layer 16             226,673,590          386.02         
Layer 17             226,673,590          386.02         
Layer 18             226,673,590          386.02         
Layer 19             226,673,590          386.02         
Layer 20             226,673,590          386.02         
Layer 21             226,673,590          386.02         
Layer 22             226,673,590          386.02         
Layer 23             226,673,590          386.02         
Layer 24             226,673,590          386.02         
Layer 25             226,673,590          386.02         
Layer 26             226,673,590          386.02         
Layer 27             226,673,590          386.02         
Layer 28             226,673,590          386.02         
Layer 29             226,673,590          386.02         
Layer 30             226,673,590          386.02         
Layer 31             226,673,590          386.02         
Final LayerNorm      229,376              0.01           
-------------------------------------------------------
Total                7,255,832,246        12602.51       

Total Inference FLOPs for 28 new tokens: 7,255,832,246
Total Memory Usage: 12602.51 MB
```

