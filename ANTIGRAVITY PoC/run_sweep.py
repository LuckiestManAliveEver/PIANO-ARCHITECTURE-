import json
import time
from antigravity_core import initialize_environment, AntiGravityContext
from pallas_sim import run_attention_simulation, SimulatedKVCache
from simulate_piano import PianoConductor

# --- Helper Functions ---

def simulate_baseline_step(instruction_len: int, query_len: int):
    """Simulates a standard Transformer run with a long system instruction."""
    ctx = AntiGravityContext.get()
    
    # 1. Input Processing
    total_tokens_in = instruction_len + query_len
    
    # Log the processing
    ctx.log_token_process(total_tokens_in)
    
    # 2. KV Cache Allocation
    # Standard Model: 32 layers, 4096 dim (approx)
    dim = 4096
    layers = 32
    kv_cache_bytes = total_tokens_in * dim * layers * 2 * 2 # (K+V) * bf16
    kv_cache_mb = kv_cache_bytes / (1024 * 1024)
    
    # 3. TTFT Simulation
    # Run attention for the full context
    run_attention_simulation(seq_len=total_tokens_in, head_dim=128, num_heads=32)
    ttft_ms = ctx.metrics.total_latency_seconds * 1000
    
    return {
        "tokens": total_tokens_in,
        "kv_mb": kv_cache_mb,
        "ttft_ms": ttft_ms
    }

def simulate_piano_step(instruction_len: int, query_len: int):
    """
    Simulates Piano Architecture.
    Hypothesis: Piano does NOT ingest the massive 'instruction_len'.
    It uses fixed specialized Keys.
    Input = Query + (Internal Key Prompts).
    Instruction_len is effectively bypassed/ignored for the Piano flow as keys are fixed.
    """
    # Reset Context for this run (mocking fresh start behavior)
    # Since AntiGravityContext is a singleton, we need to clear metrics manually or re-instantiate.
    # We will just hack the metrics reset.
    ctx = AntiGravityContext.get()
    ctx.metrics.total_flops = 0
    ctx.metrics.total_memory_bytes = 0
    ctx.metrics.total_latency_seconds = 0
    ctx.metrics.token_count = 0
    ctx.metrics.active_kernels = 0
    
    conductor = PianoConductor()
    query_mock = "x" * query_len # Just to simulate length?
    # In simulate_piano.py, query was specific.
    # Let's use the specific query length from the baseline to match.
    # Query: "If a shopkeeper..." ~ 25 tokens. 
    # Let's assume passed query_len is correct.
    
    # Execute Piano Flow (Keys K1, K2, K3)
    # K1 (30 tokens), K2 (10), K3 (10) -> Total generated/processed ~ 50.
    # Plus input query.
    
    # 1. Processing
    # Piano Input tokens = Query (Instruction is absent/distributed)
    # ctx.log_token_process(query_len) -> Done inside keys? 
    # simulate_piano.py `conduct` called `execute` which logged `avg_tokens`.
    # But it didn't explicitly log the *query* tokens in the loop initially.
    # Let's account for Query tokens manually here or update logic.
    ctx.log_token_process(query_len)
    
    # Run Conductor (which runs Keys)
    # We don't actually need to run the full string logic, just simulate the cost.
    # K1
    run_attention_simulation(seq_len=query_len + 30, head_dim=64, num_heads=8)
    ctx.log_token_process(30)
    
    # K2
    run_attention_simulation(seq_len=query_len + 30 + 10, head_dim=64, num_heads=8)
    ctx.log_token_process(10)
    
    # K3
    run_attention_simulation(seq_len=query_len + 30 + 10 + 10, head_dim=64, num_heads=8)
    ctx.log_token_process(10)
    
    total_tokens = ctx.metrics.token_count
    
    # 2. KV Cache
    # Piano KV Cache is smaller (Keys are smaller or context is smaller)
    # Max seq len ~ Query + 50.
    max_seq = query_len + 50
    kv_cache_bytes = max_seq * 4096 * 32 * 2 * 2 
    kv_cache_mb = kv_cache_bytes / (1024**2)
    
    # 3. TTFT
    # Latency of first key start (approx first attention call)
    # We accumulated all latency in ctx. 
    # TTFT is roughly first kernel latency.
    ttft_ms = (ctx.metrics.total_latency_seconds * 1000) / 3 # Rough approx
    
    return {
        "tokens": total_tokens,
        "kv_mb": kv_cache_mb,
        "ttft_ms": ttft_ms
    }

def main():
    lengths = [200, 500, 800, 1200, 2000]
    query_text = "If a shopkeeper buys 48 apples at $0.25 each and sells them at $0.40 each, what is the total profit?"
    # Simple tokenizer count
    query_len = len(query_text.split()) # Approx 25-30
    
    print(f"{'Instruction_Tokens':<20} | {'Baseline_Tokens':<15} | {'Piano_Tokens':<12} | {'Reduction_Percentage':<20}")
    print("-" * 80)
    
    for l in lengths:
        # Reset Context
        ctx = AntiGravityContext.get()
        ctx.metrics.total_flops = 0
        ctx.metrics.total_memory_bytes = 0
        ctx.metrics.total_latency_seconds = 0
        ctx.metrics.token_count = 0 
        
        # 1. Baseline
        base_res = simulate_baseline_step(l, query_len)
        
        # Reset Context
        ctx.metrics.total_flops = 0
        ctx.metrics.total_latency_seconds = 0
        ctx.metrics.token_count = 0
        
        # 2. Piano
        piano_res = simulate_piano_step(l, query_len)
        
        # 3. Calc Reduction (on Tokens In)
        # Reduction % = (Baseline - Piano) / Baseline * 100
        reduction = ((base_res['tokens'] - piano_res['tokens']) / base_res['tokens']) * 100
        
        print(f"{l:<20} | {base_res['tokens']:<15} | {piano_res['tokens']:<12} | {reduction:<20.2f}%")

if __name__ == "__main__":
    main()
