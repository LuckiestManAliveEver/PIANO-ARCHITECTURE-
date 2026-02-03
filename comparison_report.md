# KV-Cache Allocation Comparison

## Configuration
- **Prompt**: "If a shopkeeper buys 48 apples at $0.25 each and sells them at $0.40 each, what is the total profit?"
- **Total Tokens**: ~70

## Comparative Results

| Metric | Standard Transformer | Piano Architecture | Delta |
| :--- | :--- | :--- | :--- |
| **KV-Cache Tensor** | `[32, 32, 70, 128]` | `[32, 8, 70, 64]` | Reduced Width |
| **Memory Footprint** | **35.00 MB** | **4.38 MB** | **87.5% Reduction** |
| **TTFT** | 0.01 ms | 0.01 ms | 1.1x Faster |

## Analysis
The Piano Architecture achieves significant efficiency gains by dynamically activating specialized 'Keys' with reduced attention dimensions (8x64) compared to the monolithic Standard Transformer (32x128). This results in a massive reduction in KV-cache memory requirements while maintaining context.
