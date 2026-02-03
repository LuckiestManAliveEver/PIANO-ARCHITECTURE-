import time
import json
import dataclasses
from typing import List, Dict, Any
from antigravity_core import initialize_environment, AntiGravityContext
from pallas_sim import run_attention_simulation

# --- Configurations ---

@dataclasses.dataclass
class ModelConfig:
    name: str
    layers: int
    heads: int
    head_dim: int
    model_dim: int
    
    @property
    def total_kv_bytes_per_token(self):
        # 2 bytes (BF16) * 2 (K+V) * Layers * Heads * HeadDim
        return 2 * 2 * self.layers * self.heads * self.head_dim

# Standard Transformer (e.g., Llama 2 7B equivalent)
STANDARD_CONFIG = ModelConfig(
    name="Standard Transformer",
    layers=32,
    heads=32,
    head_dim=128,
    model_dim=4096
)

# Piano Architecture (Active Key Config)
# Based on simulate_piano.py: heads=8, head_dim=64
PIANO_KEY_CONFIG = ModelConfig(
    name="Piano Architecture (Key Activation)",
    layers=32, # Assuming shared depth
    heads=8,
    head_dim=64,
    model_dim=512 # 8 * 64
)

# --- Simulation Classes ---

class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.kv_shape = ""
        self.memory_mb = 0.0
        self.ttft_ms = 0.0
        self.total_tokens = 0

def run_standard_simulation(prompt: str, new_tokens: int) -> BenchmarkResult:
    ctx = AntiGravityContext.get()
    # Reset metrics for clean measurement? 
    # The context is a singleton, so we record start/end diffs.
    start_time = time.time()
    start_latency = ctx.metrics.total_latency_seconds
    
    # Simulation Parameters
    input_len = len(prompt.split())
    total_seq_len = input_len + new_tokens
    
    # 1. Prefill (Process Prompt) - This determines TTFT
    # Run attention for the prompt
    run_attention_simulation(
        seq_len=input_len, 
        head_dim=STANDARD_CONFIG.head_dim, 
        num_heads=STANDARD_CONFIG.heads
    )
    
    # 2. Decode (Generate Tokens)
    for i in range(new_tokens):
        current_seq_len = input_len + i + 1
        run_attention_simulation(
            seq_len=current_seq_len,
            head_dim=STANDARD_CONFIG.head_dim,
            num_heads=STANDARD_CONFIG.heads
        )
        ctx.log_token_process(1)

    end_latency = ctx.metrics.total_latency_seconds
    
    # Metrics Calculation
    res = BenchmarkResult(STANDARD_CONFIG.name)
    res.kv_shape = f"[{STANDARD_CONFIG.layers}, {STANDARD_CONFIG.heads}, {total_seq_len}, {STANDARD_CONFIG.head_dim}]"
    
    # Memory: Full sequence allocation
    total_kv_bytes = total_seq_len * STANDARD_CONFIG.total_kv_bytes_per_token
    res.memory_mb = total_kv_bytes / (1024 * 1024)
    
    # TTFT: Latency of the first prefill operation
    # In our simulation, we can't easily isolate the single log_kernel call time without parsing logs.
    # But we know `run_attention_simulation` adds to total_latency.
    # For simulation purposes, we'll estimate TTFT as the prefill latency.
    # Since we can't reset the singleton, we rely on the difference, but `run_attention_simulation` accumulates.
    # Let's perform a "dry run" calculation for TTFT using the same formula as pallas_sim:
    # flops = 4 * (seq_len^2) * head_dim * num_heads
    # memory = (seq_len * head_dim * num_heads * 2) * 4
    # latency = max(flops/197e12, mem/1.6e12) + 5e-6
    
    def calc_latency(s_len, h_dim, n_heads):
        flops = 4 * (s_len**2) * h_dim * n_heads
        mem = (s_len * h_dim * n_heads * 2) * 4
        return max(flops/197e12, mem/1.6e12) + 5e-6

    res.ttft_ms = calc_latency(input_len, STANDARD_CONFIG.head_dim, STANDARD_CONFIG.heads) * 1000
    res.total_tokens = total_seq_len
    
    return res

def run_piano_simulation(prompt: str, expected_key_sequence: List[str]) -> BenchmarkResult:
    ctx = AntiGravityContext.get()
    
    # Piano Mechanics:
    # 1. Intent Detection (cheap, ignored for KV comparison or assumed constant small overhead)
    # 2. Key Execution Loop
    
    current_context_len = len(prompt.split())
    total_tokens_generated = 0
    total_kv_bytes = 0
    
    # We track total keys latency for TTFT? 
    # TTFT in Piano is time to first token from the *first* Key.
    first_key_latency = 0.0
    
    for i, key_name in enumerate(expected_key_sequence):
        # Key Config (Mocked based on K1/K2/K3 from simulate_piano.py)
        # K1: 30 tokens, K2: 10, K3: 10
        if key_name == "K1":
            key_tokens = 30
        else:
            key_tokens = 10
            
        # Execute Key
        # Piano "activates" a key. The KV cache is populated for this key's execution.
        # Does Piano share KV cache across keys? 
        # If yes -> It grows like standard but with smaller width.
        # If no -> It's transient. 
        # Assumption: Shared context (autoregressive) but using the smaller "Key" dimensions.
        
        # 1. Prefill for this Key (Context so far)
        def calc_latency(s_len, h_dim, n_heads):
            flops = 4 * (s_len**2) * h_dim * n_heads
            mem = (s_len * h_dim * n_heads * 2) * 4
            return max(flops/197e12, mem/1.6e12) + 5e-6

        prefill_latency = calc_latency(current_context_len, PIANO_KEY_CONFIG.head_dim, PIANO_KEY_CONFIG.heads)
        
        if i == 0:
            first_key_latency = prefill_latency
            
        # 2. Decode for this Key
        # Simulation: Just summing up token counts for KV size
        
        current_context_len += key_tokens
        total_tokens_generated += key_tokens
        
        ctx.log_token_process(key_tokens)

    # Metrics
    res = BenchmarkResult(PIANO_KEY_CONFIG.name)
    res.kv_shape = f"[{PIANO_KEY_CONFIG.layers}, {PIANO_KEY_CONFIG.heads}, {current_context_len}, {PIANO_KEY_CONFIG.head_dim}]"
    
    # Memory: Full sequence allocation for the smaller Key config
    # Note: If Piano drops KV cache between keys (stateless), this would be much smaller (max_key_len).
    # But usually complex chains need history. We assume history is kept but in the efficient format.
    total_kv_bytes = current_context_len * PIANO_KEY_CONFIG.total_kv_bytes_per_token
    res.memory_mb = total_kv_bytes / (1024 * 1024)
    
    res.ttft_ms = first_key_latency * 1000
    res.total_tokens = current_context_len
    
    return res

# --- Main Report Generator ---

def main():
    print("Initializing Anti-Gravity Benchmark Environment...")
    initialize_environment()
    
    prompt = "If a shopkeeper buys 48 apples at $0.25 each and sells them at $0.40 each, what is the total profit?"
    # Standard: Generates equivalent of K1+K2+K3 tokens ~ 50 tokens
    tokens_to_gen = 30 + 10 + 10 
    
    print(f"\nRunning Benchmark with Prompt Length: {len(prompt.split())}, Generation: {tokens_to_gen} tokens")
    
    # 1. Standard
    std_res = run_standard_simulation(prompt, tokens_to_gen)
    
    # 2. Piano (Sequence: K1 -> K2 -> K3)
    piano_res = run_piano_simulation(prompt, ["K1", "K2", "K3"])
    
    # Report
    print("\n" + "="*60)
    print(f"{'METRIC':<25} | {'STANDARD TRANSFORMER':<20} | {'PIANO ARCHITECTURE':<20}")
    print("="*60)
    
    print(f"{'KV-Cache Shape':<25} | {std_res.kv_shape:<20} | {piano_res.kv_shape:<20}")
    print(f"{'Est. Memory (MB)':<25} | {std_res.memory_mb:<20.2f} | {piano_res.memory_mb:<20.2f}")
    print(f"{'TTFT (ms)':<25} | {std_res.ttft_ms:<20.2f} | {piano_res.ttft_ms:<20.2f}")
    print(f"{'Savings (Memory)':<25} | {'-':<20} | {100 * (1 - piano_res.memory_mb/std_res.memory_mb):.1f}%")
    print("-" * 60)
    
    # Create Comparison Report File
    report_content = f"""# KV-Cache Allocation Comparison

## Configuration
- **Prompt**: "{prompt}"
- **Total Tokens**: ~{std_res.total_tokens}

## Comparative Results

| Metric | Standard Transformer | Piano Architecture | Delta |
| :--- | :--- | :--- | :--- |
| **KV-Cache Tensor** | `{std_res.kv_shape}` | `{piano_res.kv_shape}` | Reduced Width |
| **Memory Footprint** | **{std_res.memory_mb:.2f} MB** | **{piano_res.memory_mb:.2f} MB** | **{100 * (1 - piano_res.memory_mb/std_res.memory_mb):.1f}% Reduction** |
| **TTFT** | {std_res.ttft_ms:.2f} ms | {piano_res.ttft_ms:.2f} ms | {std_res.ttft_ms / piano_res.ttft_ms:.1f}x Faster |

## Analysis
The Piano Architecture achieves significant efficiency gains by dynamically activating specialized 'Keys' with reduced attention dimensions ({PIANO_KEY_CONFIG.heads}x{PIANO_KEY_CONFIG.head_dim}) compared to the monolithic Standard Transformer ({STANDARD_CONFIG.heads}x{STANDARD_CONFIG.head_dim}). This results in a massive reduction in KV-cache memory requirements while maintaining context.
"""
    
    with open("comparison_report.md", "w") as f:
        f.write(report_content)
    
    print("\nReport saved to comparison_report.md")

if __name__ == "__main__":
    main()
