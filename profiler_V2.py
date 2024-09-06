import torch
from transformers import LlamaForCausalLM
from typing import List, Tuple
import math
import json

# calculate_inference_stats() function to calculate flops requirement

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

def analyze_layer(layer: torch.nn.Module, hidden_size: int, num_heads: int, seq_length: int) -> Tuple[int, int, Tuple[int, int]]:
    total_flops = 0
    total_params = count_parameters(layer)
    
    # Self-attention
    total_flops += estimate_flops_attention(hidden_size, num_heads, seq_length)
    
    # MLP
    intermediate_size = layer.mlp.gate_proj.out_features
    total_flops += estimate_flops_mlp(hidden_size, intermediate_size)
    
    # Layer norms (approximate)
    total_flops += 2 * hidden_size * 2  # Two layer norms per transformer layer
    
    # Output size (seq_length, hidden_size)
    output_size = (seq_length, hidden_size)
    
    return total_flops, total_params, output_size

def analyze_model(model: LlamaForCausalLM, seq_length: int, max_new_tokens: int) -> List[Tuple[str, int, int, float, Tuple[int, int]]]:
    results = []
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    
    # Embedding layer
    embed_params = count_parameters(model.model.embed_tokens)
    embed_flops = hidden_size * seq_length
    embed_memory = embed_params * 2 / (1024 * 1024)  # Assuming float16 for inference
    embed_output_size = (seq_length, hidden_size)
    results.append(("Embedding", embed_flops, embed_params, embed_memory, embed_output_size))
    
    # Transformer layers
    for i, layer in enumerate(model.model.layers):
        flops, params, output_size = analyze_layer(layer, hidden_size, num_heads, seq_length)
        memory = params * 2 / (1024 * 1024)  # Assuming float16 for inference
        results.append((f"Layer {i}", flops, params, memory, output_size))
    
    # Final layer norm
    final_ln_params = count_parameters(model.model.norm)
    final_ln_flops = hidden_size * 2
    final_ln_memory = final_ln_params * 2 / (1024 * 1024)  # Assuming float16 for inference
    final_ln_output_size = (seq_length, hidden_size)
    results.append(("Final LayerNorm", final_ln_flops, final_ln_params, final_ln_memory, final_ln_output_size))
    
    return results

def calculate_inference_stats(results: List[Tuple[str, int, int, float, Tuple[int, int]]], seq_length: int, max_new_tokens: int) -> List[Tuple[str, int, float, Tuple[int, int]]]:
    inference_stats = []
    
    for layer_name, flops, params, memory, output_size in results:
        if layer_name == "Embedding":
            inference_flops = flops
        elif layer_name == "Final LayerNorm":
            inference_flops = flops * max_new_tokens
        else:
            inference_flops = flops * (1 + 1 / seq_length)
            # inference_flops = flops * (1 + max_new_tokens / seq_length) # Keeping since ILP is considering max_tokens we are keeping accounting 1 token only
        inference_stats.append((layer_name, inference_flops, memory, output_size))
        # inference_stats.append((layer_name, flops, memory, output_size))
    
    return inference_stats

from datetime import datetime
def main():
    model_name = "meta-llama/Meta-Llama-3-8B"  
    model = LlamaForCausalLM.from_pretrained(model_name)
    seq_length = 500
    max_new_tokens = 28
    
    results = analyze_model(model, seq_length, max_new_tokens)
    inference_stats = calculate_inference_stats(results, seq_length, max_new_tokens)
    
    total_flops = 0
    total_memory = 0
    output_data = []
    
    print(f"{'Layer':<20} {'Inference FLOPs':<20} {'Memory (MB)':<15} {'Output Size':<15}")
    print("-" * 70)
    for layer_name, flops, memory, output_size in inference_stats:
        print(f"{layer_name:<20} {flops:<20,.0f} {memory:<15.2f} {output_size}")
        total_flops += flops
        total_memory += memory
        output_data.append({
            "layer": layer_name,
            "inference_flops": flops,
            "memory_mb": memory,
            "output_size": output_size
        })
    
    print("-" * 70)
    print(f"{'Total':<20} {total_flops:<20,.0f} {total_memory:<15.2f}")
    print(f"\nTotal Inference FLOPs for {max_new_tokens} new tokens: {total_flops:,.0f}")
    print(f"Total Memory Usage: {total_memory:.2f} MB")

    # Save the results to a JSON file
    timestamp=datetime.now()
    timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
    with open("inference_stats_"+timestamp+".json", "w") as f:
        json.dump({
            "model_name" : model_name,
            "seq_length" : seq_length,
            "max_new_tokens" : max_new_tokens,
            "total_flops": total_flops,
            "total_memory_mb": total_memory,
            "details": output_data
        }, f, indent=4)

main()
