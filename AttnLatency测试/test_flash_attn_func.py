import torch
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import FlashAttnFunc
import os
import pandas as pd
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Configuration
    batch_size = 32
    seq_lengths = [1024, 2048, 4096]  # Test different sequence lengths
    nheads = 32
    d = 128
    dropout_p = 0
    causal = True
    
    # Reset FlashAttnFunc's static variables
    FlashAttnFunc.latencies = {}
    FlashAttnFunc.warmup_count = 0
    FlashAttnFunc.test_count = 0
    FlashAttnFunc.current_seq_len = None
    FlashAttnFunc.save_dir = "perf_results"
    os.makedirs(FlashAttnFunc.save_dir, exist_ok=True)

    for seqlen in seq_lengths:
        print(f"\nTesting sequence length: {seqlen}")
        
        # Create input tensors
        q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)

        # Run enough iterations to cover both warmup and test phases
        # FlashAttnFunc will handle the timing internally
        for _ in range(40):  # 30 warmup + 10 test iterations
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
        # Calculate and save results
        avg_latencies = {sl: sum(lats)/len(lats) for sl, lats in FlashAttnFunc.latencies.items()}
        
        # Save to CSV
        df = pd.DataFrame({
            'seq_len': list(avg_latencies.keys()),
            'latency': list(avg_latencies.values())
        })
        csv_path = os.path.join(FlashAttnFunc.save_dir, 'flash_attn_latency.csv')
        df.to_csv(csv_path, index=False)
        