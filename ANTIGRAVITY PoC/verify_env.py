from antigravity_core import initialize_environment, AntiGravityContext
from pallas_sim import run_attention_simulation, SimulatedKVCache

def main():
    print("Starting Anti-Gravity Environment Verification...")
    
    # 1. Initialize
    ctx = initialize_environment()
    
    # 2. Simulate KV Cache
    # Batch=1, Seq=128, Dim=64, Layers=12
    kv_cache = SimulatedKVCache(max_tokens=2048, dim=64, layers=12)
    print(f"KV Cache Initialized. Total simulated size: {kv_cache.total_size} bytes")
    
    # 3. Simulate a decoding step (processing 1 token, attending to 128 previous)
    ctx.log_token_process(1)
    
    # Simulate Attention Kernel
    print("Running Pallas Attention Simulation...")
    run_attention_simulation(seq_len=128, head_dim=64, num_heads=8)
    
    # 4. Dump Metrics
    ctx.dump_metrics()
    
    # Check if metrics are non-zero
    if ctx.metrics.total_flops > 0 and ctx.metrics.active_kernels > 0:
        print("SUCCESS: Environment Verified via Simulation.")
    else:
        print("FAILURE: Simulation ran but no metrics recorded.")

if __name__ == "__main__":
    main()
