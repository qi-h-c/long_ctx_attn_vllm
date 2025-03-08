import os
import time
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils import get_max_shared_memory_bytes

# 配置参数
FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
NUM_BLOCKS = 6321  # 可以根据需要调整
PARTITION_SIZE = 512
BLOCK_SIZE = 16
HEAD_SIZE = 128
NUM_KV_HEADS = 32
NUM_QUERY_HEADS = 32
DTYPE = torch.half  # 可以是 torch.half, torch.bfloat16, 或 torch.float
KV_CACHE_DTYPE = "auto"  # 可以是 "auto" 或 "fp8"
USE_ALIBI = False

def create_test_data(num_seqs, seq_len, device):
    """创建测试数据"""
    scale = float(1.0 / (HEAD_SIZE**0.5))
    
    # 创建查询张量
    query = torch.empty(num_seqs, NUM_QUERY_HEADS, HEAD_SIZE, dtype=DTYPE, device=device)
    query.uniform_(-scale, scale)
    
    # 创建序列长度
    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int, device=device)
    
    # 创建 block tables
    max_num_blocks_per_seq = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables_lst = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)
    block_tables = torch.tensor(block_tables_lst, dtype=torch.int, device=device)
    
    # 创建 KV 缓存
    key_cache_shape = (NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE)
    value_cache_shape = (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
    
    key_cache = torch.empty(size=key_cache_shape, dtype=DTYPE, device=device)
    key_cache.uniform_(-scale, scale)
    
    value_cache = torch.empty(size=value_cache_shape, dtype=DTYPE, device=device)
    value_cache.uniform_(-scale, scale)
    
    # 创建 alibi slopes (如果需要)
    alibi_slopes = None
    if USE_ALIBI:
        alibi_slopes = torch.randn(NUM_QUERY_HEADS, dtype=torch.float, device=device)
    
    # 创建输出张量
    output = torch.empty_like(query)
    
    # 创建 KV 缓存比例因子
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    return {
        'output': output,
        'query': query,
        'key_cache': key_cache,
        'value_cache': value_cache,
        'num_kv_heads': NUM_KV_HEADS,
        'scale': scale,
        'block_tables': block_tables,
        'seq_lens': seq_lens,
        'block_size': BLOCK_SIZE,
        'max_seq_len': seq_len,
        'alibi_slopes': alibi_slopes,
        'kv_cache_dtype': KV_CACHE_DTYPE,
        'k_scale': k_scale,
        'v_scale': v_scale
    }

def test_paged_attention_time():
    """测试 paged_attention 函数的执行时间"""
    # 创建保存结果的目录
    save_dir = "perf_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 测试参数
    num_seqs = 32  # 批量大小
    seq_lengths = [1024,2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768, 34816, 36864, 38912, 40960]  # 可以根据GPU内存调整
    versions = ["v1", "v2"] if not current_platform.is_rocm() else ["v1", "v2", "rocm"]
    
    # 确保使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 存储结果
    results = {version: {} for version in versions}
    
    for version in versions:
        print(f"\n测试版本: {version}")
        
        for seq_len in seq_lengths:
            print(f"\n测试序列长度: {seq_len}")
            
            # 创建测试数据
            data = create_test_data(num_seqs, seq_len, device)
            
            # 预热阶段
            print("预热中...")
            for i in range(30):
                if version == "v1":
                    ops.paged_attention_v1(
                        data['output'],
                        data['query'],
                        data['key_cache'],
                        data['value_cache'],
                        data['num_kv_heads'],
                        data['scale'],
                        data['block_tables'],
                        data['seq_lens'],
                        data['block_size'],
                        data['max_seq_len'],
                        data['alibi_slopes'],
                        data['kv_cache_dtype'],
                        data['k_scale'],
                        data['v_scale'],
                    )
                elif version == "v2":
                    num_partitions = ((seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
                    tmp_output = torch.empty(
                        size=(num_seqs, NUM_QUERY_HEADS, num_partitions, HEAD_SIZE),
                        dtype=data['output'].dtype,
                        device=device
                    )
                    exp_sums = torch.empty(
                        size=(num_seqs, NUM_QUERY_HEADS, num_partitions),
                        dtype=torch.float32,
                        device=device
                    )
                    max_logits = torch.empty_like(exp_sums)
                    
                    ops.paged_attention_v2(
                        data['output'],
                        exp_sums,
                        max_logits,
                        tmp_output,
                        data['query'],
                        data['key_cache'],
                        data['value_cache'],
                        data['num_kv_heads'],
                        data['scale'],
                        data['block_tables'],
                        data['seq_lens'],
                        data['block_size'],
                        data['max_seq_len'],
                        data['alibi_slopes'],
                        data['kv_cache_dtype'],
                        data['k_scale'],
                        data['v_scale'],
                    )
                elif version == "rocm":
                    num_partitions = ((seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
                    tmp_output = torch.empty(
                        size=(num_seqs, NUM_QUERY_HEADS, num_partitions, HEAD_SIZE),
                        dtype=data['output'].dtype,
                        device=device
                    )
                    exp_sums = torch.empty(
                        size=(num_seqs, NUM_QUERY_HEADS, num_partitions),
                        dtype=torch.float32,
                        device=device
                    )
                    max_logits = torch.empty_like(exp_sums)
                    
                    ops.paged_attention_rocm(
                        data['output'],
                        exp_sums,
                        max_logits,
                        tmp_output,
                        data['query'],
                        data['key_cache'],
                        data['value_cache'],
                        data['num_kv_heads'],
                        data['scale'],
                        data['block_tables'],
                        data['seq_lens'],
                        data['block_size'],
                        data['max_seq_len'],
                        data['alibi_slopes'],
                        data['kv_cache_dtype'],
                        data['k_scale'],
                        data['v_scale'],
                    )
                
                torch.cuda.synchronize()
            
            # 测试阶段
            latencies = []
            print("开始测量时间...")
            for i in range(10):
                torch.cuda.synchronize()
                start_time = time.time()
                
                if version == "v1":
                    ops.paged_attention_v1(
                        data['output'],
                        data['query'],
                        data['key_cache'],
                        data['value_cache'],
                        data['num_kv_heads'],
                        data['scale'],
                        data['block_tables'],
                        data['seq_lens'],
                        data['block_size'],
                        data['max_seq_len'],
                        data['alibi_slopes'],
                        data['kv_cache_dtype'],
                        data['k_scale'],
                        data['v_scale'],
                    )
                elif version == "v2":
                    num_partitions = ((seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
                    tmp_output = torch.empty(
                        size=(num_seqs, NUM_QUERY_HEADS, num_partitions, HEAD_SIZE),
                        dtype=data['output'].dtype,
                        device=device
                    )
                    exp_sums = torch.empty(
                        size=(num_seqs, NUM_QUERY_HEADS, num_partitions),
                        dtype=torch.float32,
                        device=device
                    )
                    max_logits = torch.empty_like(exp_sums)
                    
                    ops.paged_attention_v2(
                        data['output'],
                        exp_sums,
                        max_logits,
                        tmp_output,
                        data['query'],
                        data['key_cache'],
                        data['value_cache'],
                        data['num_kv_heads'],
                        data['scale'],
                        data['block_tables'],
                        data['seq_lens'],
                        data['block_size'],
                        data['max_seq_len'],
                        data['alibi_slopes'],
                        data['kv_cache_dtype'],
                        data['k_scale'],
                        data['v_scale'],
                    )
                elif version == "rocm":
                    num_partitions = ((seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
                    tmp_output = torch.empty(
                        size=(num_seqs, NUM_QUERY_HEADS, num_partitions, HEAD_SIZE),
                        dtype=data['output'].dtype,
                        device=device
                    )
                    exp_sums = torch.empty(
                        size=(num_seqs, NUM_QUERY_HEADS, num_partitions),
                        dtype=torch.float32,
                        device=device
                    )
                    max_logits = torch.empty_like(exp_sums)
                    
                    ops.paged_attention_rocm(
                        data['output'],
                        exp_sums,
                        max_logits,
                        tmp_output,
                        data['query'],
                        data['key_cache'],
                        data['value_cache'],
                        data['num_kv_heads'],
                        data['scale'],
                        data['block_tables'],
                        data['seq_lens'],
                        data['block_size'],
                        data['max_seq_len'],
                        data['alibi_slopes'],
                        data['kv_cache_dtype'],
                        data['k_scale'],
                        data['v_scale'],
                    )
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                latency = (end_time - start_time) * 1e6  # 转换为微秒
                latencies.append(latency)
                print(f"测试 {i+1}/10: {latency:.2f} μs")
            
            # 计算平均延迟
            avg_latency = sum(latencies) / len(latencies)
            results[version][seq_len] = avg_latency
            
            # 清理内存
            del data
            torch.cuda.empty_cache()
    
    # 保存结果到CSV
    for version in versions:
        df = pd.DataFrame({
            'seq_len': list(results[version].keys()),
            'latency': list(results[version].values())
        })
        df = df.sort_values('seq_len')
        csv_path = os.path.join(save_dir, f'paged_attention_{version}_latency.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV已保存到: {csv_path}")
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.plot(df['seq_len'], df['latency'], marker='o', linewidth=2)
        plt.xlabel('序列长度')
        plt.ylabel('延迟 (μs)')
        plt.title(f'Paged Attention {version} 延迟 vs 序列长度')
        plt.grid(True)
        
        # 添加数据标签
        for x, y in zip(df['seq_len'], df['latency']):
            plt.annotate(f'{y:.2f}', 
                        (x, y), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        # 保存图表
        plt.tight_layout()
        png_path = os.path.join(save_dir, f'paged_attention_{version}_latency.png')
        plt.savefig(png_path)
        plt.close()
    
    # 创建比较图表
    plt.figure(figsize=(12, 7))
    for version in versions:
        df = pd.DataFrame({
            'seq_len': list(results[version].keys()),
            'latency': list(results[version].values())
        })
        df = df.sort_values('seq_len')
        plt.plot(df['seq_len'], df['latency'], marker='o', linewidth=2, label=f'Version {version}')
    
    plt.xlabel('序列长度')
    plt.ylabel('延迟 (μs)')
    plt.title('Paged Attention 不同版本延迟比较')
    plt.grid(True)
    plt.legend()
    
    # 保存比较图表
    plt.tight_layout()
    comparison_png_path = os.path.join(save_dir, 'paged_attention_comparison.png')
    plt.savefig(comparison_png_path)
    plt.close()
    
    # 打印表格形式的数据
    print("\n延迟结果:")
    print("-" * 70)
    header = f"{'序列长度':^15}"
    for version in versions:
        header += f" | {f'版本 {version} (μs)':^15}"
    print(header)
    print("-" * 70)
    
    for seq_len in seq_lengths:
        row = f"{seq_len:^15}"
        for version in versions:
            row += f" | {results[version][seq_len]:^15.2f}"
        print(row)
    print("-" * 70)
    
    print(f"\n测试完成")
    print(f"结果已保存到: {save_dir}")

if __name__ == "__main__":
    test_paged_attention_time()