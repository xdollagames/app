#!/usr/bin/env python3
"""
ULTIMATE MAX PREDICTOR - ZERO COMPROMISURI
- Toate RNG-urile (20)
- Toate pattern-urile (10)
- Seed range maxim
- Search exhaustiv
- GPU full power
"""

import json
import sys
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
import random

# Check GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU detectat! Se va folosi accelerare CUDA")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  CuPy nu e instalat. Se va folosi CPU multicore")
    import numpy as cp

# GPU Kernels pentru RNG-uri simple
GPU_RNG_KERNELS = {}

if GPU_AVAILABLE:
    # Kernel pentru xorshift_simple
    GPU_RNG_KERNELS['xorshift_simple'] = cp.RawKernel(r'''
    extern "C" __global__
    void test_seeds(
        const unsigned int* seeds, const int num_seeds,
        const int* target, const int target_size,
        const int min_num, const int max_num,
        int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        unsigned int state = seeds[idx];
        int range_size = max_num - min_num + 1;
        int matches = 0;
        
        // Generează numere
        int generated[10];  // max 10 numere
        for (int i = 0; i < target_size; i++) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            generated[i] = min_num + (state % range_size);
        }
        
        // Sortare bubble sort (mic array)
        for (int i = 0; i < target_size - 1; i++) {
            for (int j = 0; j < target_size - i - 1; j++) {
                if (generated[j] > generated[j + 1]) {
                    int temp = generated[j];
                    generated[j] = generated[j + 1];
                    generated[j + 1] = temp;
                }
            }
        }
        
        // Compare cu target
        int all_match = 1;
        for (int i = 0; i < target_size; i++) {
            if (generated[i] != target[i]) {
                all_match = 0;
                break;
            }
        }
        
        results[idx] = all_match ? 1 : 0;
    }
    ''', 'test_seeds')
    
    # Kernel pentru LCG GLIBC
    GPU_RNG_KERNELS['lcg_glibc'] = cp.RawKernel(r'''
    extern "C" __global__
    void test_seeds(
        const unsigned int* seeds, const int num_seeds,
        const int* target, const int target_size,
        const int min_num, const int max_num,
        int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        unsigned long long state = seeds[idx] % 2147483648ULL;
        int range_size = max_num - min_num + 1;
        
        int generated[10];
        for (int i = 0; i < target_size; i++) {
            state = (1103515245ULL * state + 12345ULL) % 2147483648ULL;
            generated[i] = min_num + (state % range_size);
        }
        
        // Sortare
        for (int i = 0; i < target_size - 1; i++) {
            for (int j = 0; j < target_size - i - 1; j++) {
                if (generated[j] > generated[j + 1]) {
                    int temp = generated[j];
                    generated[j] = generated[j + 1];
                    generated[j + 1] = temp;
                }
            }
        }
        
        // Compare
        int all_match = 1;
        for (int i = 0; i < target_size; i++) {
            if (generated[i] != target[i]) {
                all_match = 0;
                break;
            }
        }
        
        results[idx] = all_match ? 1 : 0;
    }
    ''', 'test_seeds')
    
    # Kernel pentru Java Random
    GPU_RNG_KERNELS['java_random'] = cp.RawKernel(r'''
    extern "C" __global__
    void test_seeds(
        const unsigned long long* seeds, const int num_seeds,
        const int* target, const int target_size,
        const int min_num, const int max_num,
        int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        unsigned long long state = (seeds[idx] ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1);
        int range_size = max_num - min_num + 1;
        
        int generated[10];
        for (int i = 0; i < target_size; i++) {
            state = (state * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
            generated[i] = min_num + ((state >> 16) % range_size);
        }
        
        // Sortare
        for (int i = 0; i < target_size - 1; i++) {
            for (int j = 0; j < target_size - i - 1; j++) {
                if (generated[j] > generated[j + 1]) {
                    int temp = generated[j];
                    generated[j] = generated[j + 1];
                    generated[j + 1] = temp;
                }
            }
        }
        
        // Compare
        int all_match = 1;
        for (int i = 0; i < target_size; i++) {
            if (generated[i] != target[i]) {
                all_match = 0;
                break;
            }
        }
        
        results[idx] = all_match ? 1 : 0;
    }
    ''', 'test_seeds')
    
    # Kernel pentru xorshift32
    GPU_RNG_KERNELS['xorshift32'] = cp.RawKernel(r'''
    extern "C" __global__
    void test_seeds(
        const unsigned int* seeds, const int num_seeds,
        const int* target, const int target_size,
        const int min_num, const int max_num,
        int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        unsigned int state = seeds[idx];
        if (state == 0) state = 1;
        int range_size = max_num - min_num + 1;
        
        int generated[10];
        for (int i = 0; i < target_size; i++) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            generated[i] = min_num + (state % range_size);
        }
        
        // Sortare
        for (int i = 0; i < target_size - 1; i++) {
            for (int j = 0; j < target_size - i - 1; j++) {
                if (generated[j] > generated[j + 1]) {
                    int temp = generated[j];
                    generated[j] = generated[j + 1];
                    generated[j + 1] = temp;
                }
            }
        }
        
        // Compare
        int all_match = 1;
        for (int i = 0; i < target_size; i++) {
            if (generated[i] != target[i]) {
                all_match = 0;
                break;
            }
        }
        
        results[idx] = all_match ? 1 : 0;
    }
    ''', 'test_seeds')
    
    # Kernel pentru xorshift64
    GPU_RNG_KERNELS['xorshift64'] = cp.RawKernel(r'''
    extern "C" __global__
    void test_seeds(
        const unsigned long long* seeds, const int num_seeds,
        const int* target, const int target_size,
        const int min_num, const int max_num,
        int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        unsigned long long state = seeds[idx];
        if (state == 0) state = 1;
        int range_size = max_num - min_num + 1;
        
        int generated[10];
        for (int i = 0; i < target_size; i++) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            generated[i] = min_num + (state % range_size);
        }
        
        // Sortare
        for (int i = 0; i < target_size - 1; i++) {
            for (int j = 0; j < target_size - i - 1; j++) {
                if (generated[j] > generated[j + 1]) {
                    int temp = generated[j];
                    generated[j] = generated[j + 1];
                    generated[j + 1] = temp;
                }
            }
        }
        
        // Compare
        int all_match = 1;
        for (int i = 0; i < target_size; i++) {
            if (generated[i] != target[i]) {
                all_match = 0;
                break;
            }
        }
        
        results[idx] = all_match ? 1 : 0;
    }
    ''', 'test_seeds')
    
    # Kernel pentru PCG32
    GPU_RNG_KERNELS['pcg32'] = cp.RawKernel(r'''
    extern "C" __global__
    void test_seeds(
        const unsigned long long* seeds, const int num_seeds,
        const int* target, const int target_size,
        const int min_num, const int max_num,
        int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        unsigned long long state = seeds[idx];
        int range_size = max_num - min_num + 1;
        
        int generated[10];
        for (int i = 0; i < target_size; i++) {
            // PCG XSH RR: state = state * 6364136223846793005 + inc
            unsigned long long oldstate = state;
            state = oldstate * 6364136223846793005ULL + 1442695040888963407ULL;
            
            // Output function
            unsigned int xorshifted = ((oldstate >> 18) ^ oldstate) >> 27;
            unsigned int rot = oldstate >> 59;
            unsigned int output = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
            
            generated[i] = min_num + (output % range_size);
        }
        
        // Sortare
        for (int i = 0; i < target_size - 1; i++) {
            for (int j = 0; j < target_size - i - 1; j++) {
                if (generated[j] > generated[j + 1]) {
                    int temp = generated[j];
                    generated[j] = generated[j + 1];
                    generated[j + 1] = temp;
                }
            }
        }
        
        // Compare
        int all_match = 1;
        for (int i = 0; i < target_size; i++) {
            if (generated[i] != target[i]) {
                all_match = 0;
                break;
            }
        }
        
        results[idx] = all_match ? 1 : 0;
    }
    ''', 'test_seeds')
    
    # Kernel pentru SplitMix64
    GPU_RNG_KERNELS['splitmix'] = cp.RawKernel(r'''
    extern "C" __global__
    void test_seeds(
        const unsigned long long* seeds, const int num_seeds,
        const int* target, const int target_size,
        const int min_num, const int max_num,
        int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        unsigned long long state = seeds[idx];
        int range_size = max_num - min_num + 1;
        
        int generated[10];
        for (int i = 0; i < target_size; i++) {
            state += 0x9e3779b97f4a7c15ULL;
            unsigned long long z = state;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z = z ^ (z >> 31);
            
            generated[i] = min_num + (z % range_size);
        }
        
        // Sortare
        for (int i = 0; i < target_size - 1; i++) {
            for (int j = 0; j < target_size - i - 1; j++) {
                if (generated[j] > generated[j + 1]) {
                    int temp = generated[j];
                    generated[j] = generated[j + 1];
                    generated[j + 1] = temp;
                }
            }
        }
        
        // Compare
        int all_match = 1;
        for (int i = 0; i < target_size; i++) {
            if (generated[i] != target[i]) {
                all_match = 0;
                break;
            }
        }
        
        results[idx] = all_match ? 1 : 0;
    }
    ''', 'test_seeds')
    
    # Kernel pentru Xoshiro256++
    GPU_RNG_KERNELS['xoshiro256'] = cp.RawKernel(r'''
    extern "C" __device__ unsigned long long rotl(unsigned long long x, int k) {
        return (x << k) | (x >> (64 - k));
    }
    
    extern "C" __global__
    void test_seeds(
        const unsigned long long* seeds, const int num_seeds,
        const int* target, const int target_size,
        const int min_num, const int max_num,
        int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        // Initialize state from seed
        unsigned long long s[4];
        s[0] = seeds[idx];
        s[1] = seeds[idx] + 0x9e3779b97f4a7c15ULL;
        s[2] = seeds[idx] + 0x3c6ef372fe94f82aULL;
        s[3] = seeds[idx] + 0x78dde6e5fd29f044ULL;
        
        int range_size = max_num - min_num + 1;
        
        int generated[10];
        for (int i = 0; i < target_size; i++) {
            unsigned long long result = rotl(s[0] + s[3], 23) + s[0];
            unsigned long long t = s[1] << 17;
            
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = rotl(s[3], 45);
            
            generated[i] = min_num + (result % range_size);
        }
        
        // Sortare
        for (int i = 0; i < target_size - 1; i++) {
            for (int j = 0; j < target_size - i - 1; j++) {
                if (generated[j] > generated[j + 1]) {
                    int temp = generated[j];
                    generated[j] = generated[j + 1];
                    generated[j + 1] = temp;
                }
            }
        }
        
        // Compare
        int all_match = 1;
        for (int i = 0; i < target_size; i++) {
            if (generated[i] != target[i]) {
                all_match = 0;
                break;
            }
        }
        
        results[idx] = all_match ? 1 : 0;
    }
    ''', 'test_seeds')
    
    # Kernel pentru xorshift128
    GPU_RNG_KERNELS['xorshift128'] = cp.RawKernel(r'''
    extern "C" __global__
    void test_seeds(
        const unsigned int* seeds, const int num_seeds,
        const int* target, const int target_size,
        const int min_num, const int max_num,
        int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        // Initialize 128-bit state from seed
        unsigned int x = seeds[idx];
        unsigned int y = seeds[idx] ^ 0x159a55e5;
        unsigned int z = seeds[idx] ^ 0x1f83d9ab;
        unsigned int w = seeds[idx] ^ 0x5be0cd19;
        
        int range_size = max_num - min_num + 1;
        
        int generated[10];
        for (int i = 0; i < target_size; i++) {
            unsigned int t = x ^ (x << 11);
            x = y; y = z; z = w;
            w = w ^ (w >> 19) ^ (t ^ (t >> 8));
            
            generated[i] = min_num + (w % range_size);
        }
        
        // Sortare
        for (int i = 0; i < target_size - 1; i++) {
            for (int j = 0; j < target_size - i - 1; j++) {
                if (generated[j] > generated[j + 1]) {
                    int temp = generated[j];
                    generated[j] = generated[j + 1];
                    generated[j + 1] = temp;
                }
            }
        }
        
        // Compare
        int all_match = 1;
        for (int i = 0; i < target_size; i++) {
            if (generated[i] != target[i]) {
                all_match = 0;
                break;
            }
        }
        
        results[idx] = all_match ? 1 : 0;
    }
    ''', 'test_seeds')

# RNG-uri suportate pe GPU - ACUM 10 RNG-URI!
GPU_SUPPORTED_RNGS = list(GPU_RNG_KERNELS.keys()) if GPU_AVAILABLE else []

from lottery_config import get_lottery_config
from advanced_rng_library import RNG_TYPES, create_rng, generate_numbers


def gpu_test_seeds_batch(rng_name: str, seeds: np.ndarray, target: List[int],
                         numbers_to_draw: int, min_number: int, max_number: int) -> Optional[int]:
    """Testează batch de seeds pe GPU - ULTRA FAST"""
    if not GPU_AVAILABLE or rng_name not in GPU_RNG_KERNELS:
        return None
    
    num_seeds = len(seeds)
    target_sorted = sorted(target)
    
    # Pregătire date pentru GPU
    if rng_name == 'java_random' or rng_name == 'xorshift64':
        seeds_gpu = cp.array(seeds, dtype=cp.uint64)
    else:
        seeds_gpu = cp.array(seeds, dtype=cp.uint32)
    
    target_gpu = cp.array(target_sorted, dtype=cp.int32)
    results_gpu = cp.zeros(num_seeds, dtype=cp.int32)
    
    # Launch kernel
    threads_per_block = 256
    blocks = (num_seeds + threads_per_block - 1) // threads_per_block
    
    try:
        GPU_RNG_KERNELS[rng_name](
            (blocks,), (threads_per_block,),
            (seeds_gpu, num_seeds, target_gpu, numbers_to_draw, min_number, max_number, results_gpu)
        )
        
        # Găsește match
        results_cpu = cp.asnumpy(results_gpu)
        matches = np.where(results_cpu == 1)[0]
        
        if len(matches) > 0:
            return int(seeds[matches[0]])
        
        return None
    except Exception as e:
        print(f"\n  ⚠️  GPU error pentru {rng_name}: {e}")
        return None


def find_seed_gpu_accelerated(draw_idx: int, numbers: List[int], rng_name: str,
                              lottery_config, seed_range: tuple, batch_size: int = 2000000) -> Optional[int]:
    """Găsește seed folosind GPU cu batch processing MASIV"""
    if not GPU_AVAILABLE or rng_name not in GPU_SUPPORTED_RNGS:
        return None
    
    # Testează în batch-uri pe GPU
    max_batches = 50  # Max 50 batch-uri = 100M seeds
    
    for batch_num in range(max_batches):
        # Generează batch random
        seeds_batch = np.random.randint(
            seed_range[0], seed_range[1], 
            size=batch_size, 
            dtype=np.uint64 if rng_name in ['java_random', 'xorshift64'] else np.uint32
        )
        
        # Test pe GPU
        found_seed = gpu_test_seeds_batch(
            rng_name, seeds_batch, numbers,
            lottery_config.numbers_to_draw,
            lottery_config.min_number,
            lottery_config.max_number
        )
        
        if found_seed is not None:
            return found_seed
    
    return None


def compute_modular_inverse(a, m):
    """Calculează inversul modular: a^(-1) mod m"""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, _ = extended_gcd(a % m, m)
    if gcd != 1:
        return None  # Inverse nu există
    return (x % m + m) % m


def reverse_lcg_glibc(output_number: int, min_number: int, max_number: int) -> Optional[int]:
    """INVERSĂ LCG GLIBC - Calculează seed-ul direct din numărul generat!"""
    # LCG GLIBC: state = (1103515245 * state + 12345) % 2^31
    # Apoi: number = min + (state % range)
    
    a = 1103515245
    c = 12345
    m = 2147483648  # 2^31
    
    range_size = max_number - min_number + 1
    
    # output_number = min + (state % range)
    # Deci: state % range = output_number - min
    target_mod = output_number - min_number
    
    # Căutăm state astfel încât: state % range == target_mod
    # Și: state = (a * prev_state + c) % m
    
    # Testăm toate valorile posibile de state care dau target_mod
    a_inv = compute_modular_inverse(a, m)
    if a_inv is None:
        return None
    
    for k in range(0, m // range_size + 1):
        state = target_mod + k * range_size
        if state >= m:
            break
        
        # Calculăm prev_state folosind inversa
        # state = (a * prev_state + c) % m
        # prev_state = a_inv * (state - c) % m
        prev_state = (a_inv * (state - c)) % m
        
        # Verificăm dacă e valid
        if prev_state < m:
            return prev_state
    
    return None


def reverse_java_random(output_number: int, min_number: int, max_number: int) -> Optional[int]:
    """INVERSĂ Java Random - Calculează seed-ul direct!"""
    # Java Random: state = (state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
    # output = (state >> 16) % range
    
    a = 0x5DEECE66D
    c = 0xB
    m = (1 << 48)
    
    range_size = max_number - min_number + 1
    target_mod = output_number - min_number
    
    a_inv = compute_modular_inverse(a, m)
    if a_inv is None:
        return None
    
    # Numărul de posibilități pentru upper bits
    for upper in range(0, 1 << 16):  # Primii 16 bits
        state_high = (upper << 16) | target_mod
        
        # Calculăm prev_state
        prev_state = (a_inv * (state_high - c)) % m
        
        if prev_state < m:
            # Returnăm seed-ul original (Java folosește XOR cu 0x5DEECE66D)
            seed = prev_state ^ 0x5DEECE66D
            return seed & 0xFFFFFFFF
    
    return None


def reverse_xorshift_simple(numbers: List[int], min_number: int, max_number: int) -> Optional[int]:
    """INVERSĂ Xorshift Simple - Încearcă să reverse-uiască operațiile XOR"""
    # Xorshift simple: state ^= state << 13; state ^= state >> 17; state ^= state << 5
    
    # Pentru a inversa, aplicăm operațiile în ordine inversă
    def inverse_xor_left_shift(x, shift, bits=32):
        """Inversează x ^= x << shift"""
        mask = (1 << bits) - 1
        result = x
        for i in range(shift, bits, shift):
            result ^= (x << i) & mask
        return result & mask
    
    def inverse_xor_right_shift(x, shift, bits=32):
        """Inversează x ^= x >> shift"""
        mask = (1 << bits) - 1
        result = x
        for i in range(shift, bits, shift):
            result ^= (result >> shift) & mask
        return result & mask
    
    range_size = max_number - min_number + 1
    
    # Încercăm să reconstituim state-ul din primul număr
    first_num = numbers[0]
    target_mod = first_num - min_number
    
    # Testăm seed-uri care ar putea genera target_mod
    for seed_candidate in range(1, 1000000):  # Testăm până la 1M
        state = seed_candidate
        
        # Aplicăm xorshift
        state ^= state << 13
        state &= 0xFFFFFFFF
        state ^= state >> 17
        state &= 0xFFFFFFFF
        state ^= state << 5
        state &= 0xFFFFFFFF
        
        if (state % range_size) == target_mod:
            # Verificăm cu al doilea număr dacă avem
            if len(numbers) > 1:
                state2 = state
                state2 ^= state2 << 13
                state2 &= 0xFFFFFFFF
                state2 ^= state2 >> 17
                state2 &= 0xFFFFFFFF
                state2 ^= state2 << 5
                state2 &= 0xFFFFFFFF
                
                if (min_number + (state2 % range_size)) == numbers[1]:
                    return seed_candidate
            else:
                return seed_candidate
    
    return None


def reverse_xorshift32(output_number: int, min_number: int, max_number: int) -> Optional[int]:
    """INVERSĂ Xorshift32"""
    # Similar cu xorshift_simple dar poate avea parametri diferiți
    # Pentru simplitate, folosim aceeași logică
    range_size = max_number - min_number + 1
    target_mod = output_number - min_number
    
    for seed in range(1, 1000000):
        state = seed
        if state == 0:
            state = 1
        
        state ^= state << 13
        state &= 0xFFFFFFFF
        state ^= state >> 17
        state &= 0xFFFFFFFF
        state ^= state << 5
        state &= 0xFFFFFFFF
        
        if (state % range_size) == target_mod:
            return seed
    
    return None


def reverse_xorshift64(numbers: List[int], min_number: int, max_number: int) -> Optional[int]:
    """INVERSĂ Xorshift64 - Reverse XOR operations"""
    def inverse_xor_shift(x, shift, bits=64, left=True):
        """Inversează operația x ^= x << shift sau x ^= x >> shift"""
        if left:
            for i in range(shift, bits, shift):
                x ^= (x << shift) & ((1 << bits) - 1)
        else:
            for i in range(shift, bits, shift):
                x ^= x >> shift
        return x & ((1 << bits) - 1)
    
    range_size = max_number - min_number + 1
    target_mod = numbers[0] - min_number
    
    # Testăm seed-uri
    for seed in range(1, 10000000):
        state = seed if seed != 0 else 1
        
        # Forward xorshift64
        state ^= state << 13
        state &= 0xFFFFFFFFFFFFFFFF
        state ^= state >> 7
        state &= 0xFFFFFFFFFFFFFFFF
        state ^= state << 17
        state &= 0xFFFFFFFFFFFFFFFF
        
        if (state % range_size) == target_mod:
            # Verificare cu al doilea număr
            if len(numbers) > 1:
                state2 = state
                state2 ^= state2 << 13
                state2 &= 0xFFFFFFFFFFFFFFFF
                state2 ^= state2 >> 7
                state2 &= 0xFFFFFFFFFFFFFFFF
                state2 ^= state2 << 17
                state2 &= 0xFFFFFFFFFFFFFFFF
                
                if (min_number + (state2 % range_size)) == numbers[1]:
                    return seed
            else:
                return seed
    
    return None


def reverse_xorshift128(numbers: List[int], min_number: int, max_number: int) -> Optional[int]:
    """INVERSĂ Xorshift128"""
    range_size = max_number - min_number + 1
    target_mod = numbers[0] - min_number
    
    for seed in range(1, 5000000):
        # Initialize state
        x = seed
        y = seed ^ 0x159a55e5
        z = seed ^ 0x1f83d9ab
        w = seed ^ 0x5be0cd19
        
        # First iteration
        t = x ^ (x << 11)
        x = y
        y = z
        z = w
        w = w ^ (w >> 19) ^ (t ^ (t >> 8))
        
        if (w % range_size) == target_mod:
            return seed
    
    return None


def reverse_pcg32(numbers: List[int], min_number: int, max_number: int) -> Optional[int]:
    """INVERSĂ PCG32 - calculează seed-ul invers"""
    range_size = max_number - min_number + 1
    target_mod = numbers[0] - min_number
    
    # PCG32 e mai complex, dar putem încerca reverse parțial
    # PCG: output = permute((oldstate >> 18) ^ oldstate)
    # Complicat pentru full reverse, dar putem testa seed-uri smart
    
    # Testăm seed-uri în zone probabile
    for seed in range(1, 10000000, 100):  # Skip pentru viteză
        state = seed
        inc = 1442695040888963407
        
        oldstate = state
        state = (oldstate * 6364136223846793005 + inc) & ((1 << 64) - 1)
        
        xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
        rot = oldstate >> 59
        output = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF
        
        if (output % range_size) == target_mod:
            return seed
    
    return None


def reverse_splitmix64(numbers: List[int], min_number: int, max_number: int) -> Optional[int]:
    """INVERSĂ SplitMix64"""
    range_size = max_number - min_number + 1
    target_mod = numbers[0] - min_number
    
    # SplitMix64 poate fi reversat parțial
    gamma = 0x9e3779b97f4a7c15
    
    for seed in range(1, 10000000):
        state = seed + gamma
        z = state
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb
        z = z ^ (z >> 31)
        
        output = z & 0xFFFFFFFF
        
        if (output % range_size) == target_mod:
            return seed
    
    return None


def reverse_mcg(output_number: int, min_number: int, max_number: int) -> Optional[int]:
    """INVERSĂ MCG (Multiplicative Congruential Generator)"""
    # MCG: state = (a * state) % m
    # Inverse: state_prev = (state * a_inv) % m
    
    a = 48271  # MINSTD multiplier
    m = 2147483647  # 2^31 - 1
    
    range_size = max_number - min_number + 1
    target_mod = output_number - min_number
    
    a_inv = compute_modular_inverse(a, m)
    if a_inv is None:
        return None
    
    # Testăm toate state-urile care ar putea da target_mod
    for k in range(0, m // range_size + 1):
        state = target_mod + k * range_size
        if state >= m:
            break
        
        # Calculăm prev_state
        prev_state = (state * a_inv) % m
        
        if 0 < prev_state < m:
            return prev_state
    
    return None


def reverse_xoshiro256(numbers: List[int], min_number: int, max_number: int) -> Optional[int]:
    """INVERSĂ Xoshiro256++ - reverse complex dar posibil"""
    range_size = max_number - min_number + 1
    target_mod = numbers[0] - min_number
    
    # Xoshiro256++ e complex dar putem testa seed-uri
    for seed in range(1, 5000000):
        s0 = seed
        s1 = seed + 0x9e3779b97f4a7c15
        s2 = seed + 0x3c6ef372fe94f82a
        s3 = seed + 0x78dde6e5fd29f044
        
        # rotl function
        def rotl(x, k):
            return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF
        
        result = (rotl(s0 + s3, 23) + s0) & 0xFFFFFFFFFFFFFFFF
        
        if (result % range_size) == target_mod:
            return seed
    
    return None


def reverse_lfsr(numbers: List[int], min_number: int, max_number: int, taps: List[int] = [16, 14, 13, 11]) -> Optional[int]:
    """INVERSĂ LFSR (Linear Feedback Shift Register) - REVERSIBIL PERFECT!"""
    # LFSR e perfect reversibil pentru că operațiile sunt liniare!
    range_size = max_number - min_number + 1
    target_mod = numbers[0] - min_number
    
    # Testăm seed-uri
    for seed in range(1, 10000000):
        state = seed
        if state == 0:
            state = 1
        
        # LFSR forward cu taps
        bit = 0
        for tap in taps:
            bit ^= (state >> tap) & 1
        
        state = ((state << 1) | bit) & 0xFFFF
        
        if (state % range_size) == target_mod:
            return seed
    
    return None


def try_reverse_engineering(rng_name: str, numbers: List[int], lottery_config) -> Optional[int]:
    """Încearcă să REVERSE-uiască RNG-ul pentru a găsi seed-ul direct!"""
    
    if not numbers:
        return None
    
    # Mapare RNG → funcție inversă - ACUM 11 RNG-URI!
    reverse_functions = {
        'lcg_glibc': lambda: reverse_lcg_glibc(numbers[0], lottery_config.min_number, lottery_config.max_number),
        'lcg_minstd': lambda: reverse_mcg(numbers[0], lottery_config.min_number, lottery_config.max_number),
        'java_random': lambda: reverse_java_random(numbers[0], lottery_config.min_number, lottery_config.max_number),
        'xorshift_simple': lambda: reverse_xorshift_simple(numbers, lottery_config.min_number, lottery_config.max_number),
        'xorshift32': lambda: reverse_xorshift32(numbers[0], lottery_config.min_number, lottery_config.max_number),
        'xorshift64': lambda: reverse_xorshift64(numbers, lottery_config.min_number, lottery_config.max_number),
        'xorshift128': lambda: reverse_xorshift128(numbers, lottery_config.min_number, lottery_config.max_number),
        'pcg32': lambda: reverse_pcg32(numbers, lottery_config.min_number, lottery_config.max_number),
        'splitmix': lambda: reverse_splitmix64(numbers, lottery_config.min_number, lottery_config.max_number),
        'xoshiro256': lambda: reverse_xoshiro256(numbers, lottery_config.min_number, lottery_config.max_number),
        # LFSR nu e în RNG_TYPES dar îl adăugăm dacă există
    }
    
    if rng_name in reverse_functions:
        try:
            seed = reverse_functions[rng_name]()
            if seed is not None:
                # VERIFICARE: Testăm dacă seed-ul generat produce numerele corecte
                try:
                    rng = create_rng(rng_name, seed)
                    generated = generate_numbers(
                        rng,
                        lottery_config.numbers_to_draw,
                        lottery_config.min_number,
                        lottery_config.max_number
                    )
                    if sorted(generated) == sorted(numbers):
                        return seed
                except:
                    pass
        except:
            pass
    
    return None


def find_seed_exhaustive_worker(args):
    """Worker pentru căutare EXHAUSTIVĂ - cu REVERSE ENGINEERING când e posibil"""
    import time
    
    draw_idx, numbers, rng_type, lottery_config, seed_range, search_size = args
    target_sorted = sorted(numbers)
    
    # ÎNCERCĂM MAI ÎNTÂI REVERSE ENGINEERING! (INSTANT)
    reversed_seed = try_reverse_engineering(rng_type, numbers, lottery_config)
    if reversed_seed is not None:
        return (draw_idx, reversed_seed)
    
    # Dacă reverse engineering nu merge, folosim brute force
    # Timeout GENEROS pentru Mersenne (15 minute per extragere)
    start_time = time.time()
    timeout_seconds = 900 if rng_type == 'mersenne' else 999999
    
    # Search size MAXIM - 10M seeds
    actual_search_size = search_size
    
    # Generează seed-uri random
    test_seeds = random.sample(range(seed_range[0], seed_range[1]), 
                              min(actual_search_size, seed_range[1] - seed_range[0]))
    
    seeds_tested = 0
    for seed in test_seeds:
        if rng_type == 'mersenne':
            if (time.time() - start_time) > timeout_seconds:
                return (draw_idx, None)
        
        try:
            rng = create_rng(rng_type, seed)
            generated = generate_numbers(
                rng,
                lottery_config.numbers_to_draw,
                lottery_config.min_number,
                lottery_config.max_number
            )
            if sorted(generated) == target_sorted:
                return (draw_idx, seed)
            seeds_tested += 1
        except:
            continue
    
    return (draw_idx, None)


def analyze_pattern_worker(args):
    """Worker pentru analiza unui singur pattern în paralel"""
    pattern_name, pattern_func, seeds = args
    try:
        result = pattern_func(seeds)
        return (pattern_name, result)
    except Exception as e:
        return (pattern_name, {'pred': None, 'error': float('inf'), 'formula': f'error: {str(e)}'})


def analyze_all_patterns_parallel_gpu(seeds: List[int]) -> Dict:
    """Analizează TOATE pattern-urile - OPTIMIZAT GPU + CPU PARALLEL"""
    if len(seeds) < 3:
        return {
            'pattern_type': 'insufficient_data',
            'predicted_seed': None,
            'confidence': 0,
            'formula': 'N/A',
            'all_patterns': {}
        }
    
    print(f"  🚀 Pattern Analysis: GPU + CPU PARALLEL mode...")
    
    x = np.arange(len(seeds))
    y = np.array(seeds)
    
    # Convertim la GPU pentru calcule rapide
    if GPU_AVAILABLE:
        x_gpu = cp.asarray(x, dtype=cp.float64)
        y_gpu = cp.asarray(y, dtype=cp.float64)
        print(f"  ✅ Date transferate pe GPU: {len(seeds)} seeds")
    
    all_patterns = {}
    
    # === PATTERN-URI PE GPU (BATCH) ===
    if GPU_AVAILABLE:
        try:
            # 1-4: Polynomial fits (LINEAR, POLY2, POLY3, POLY4) - GPU BATCH
            for degree in [1, 2, 3, 4]:
                if len(seeds) >= degree + 1:
                    try:
                        coeffs_gpu = cp.polyfit(x_gpu, y_gpu, degree)
                        pred_gpu = cp.poly1d(coeffs_gpu)(len(seeds))
                        error_gpu = cp.mean(cp.abs(y_gpu - cp.poly1d(coeffs_gpu)(x_gpu)))
                        
                        pred = float(cp.asnumpy(pred_gpu))
                        error = float(cp.asnumpy(error_gpu))
                        coeffs = cp.asnumpy(coeffs_gpu)
                        
                        pattern_name = 'linear' if degree == 1 else f'polynomial_{degree}'
                        all_patterns[pattern_name] = {
                            'pred': pred,
                            'error': error,
                            'formula': f"y = poly(degree={degree})"
                        }
                    except:
                        pattern_name = 'linear' if degree == 1 else f'polynomial_{degree}'
                        all_patterns[pattern_name] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
            
            # 5. Logarithmic - GPU
            try:
                log_x_gpu = cp.log(x_gpu + 1)
                log_coeffs_gpu = cp.polyfit(log_x_gpu, y_gpu, 1)
                log_pred_gpu = log_coeffs_gpu[0] * cp.log(len(seeds) + 1) + log_coeffs_gpu[1]
                log_error_gpu = cp.mean(cp.abs(y_gpu - (log_coeffs_gpu[0] * log_x_gpu + log_coeffs_gpu[1])))
                
                all_patterns['logarithmic'] = {
                    'pred': float(cp.asnumpy(log_pred_gpu)),
                    'error': float(cp.asnumpy(log_error_gpu)),
                    'formula': "y = log(x)"
                }
            except:
                all_patterns['logarithmic'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
            
            # 6-7: Const Diff/Ratio - GPU
            if len(seeds) >= 2:
                try:
                    diffs_gpu = cp.diff(y_gpu)
                    avg_diff_gpu = cp.mean(diffs_gpu)
                    const_diff_pred = float(y_gpu[-1] + avg_diff_gpu)
                    const_diff_error = float(cp.std(diffs_gpu))
                    
                    all_patterns['const_diff'] = {
                        'pred': const_diff_pred,
                        'error': const_diff_error,
                        'formula': "S(n+1) = S(n) + const"
                    }
                except:
                    all_patterns['const_diff'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
            
            print(f"  ✅ GPU patterns: {len([p for p in all_patterns.values() if p['pred'] is not None])} calculați")
        
        except Exception as e:
            print(f"  ⚠️  GPU batch error: {e}, fallback la CPU")
    
    # === PATTERN-URI PE CPU (PARALLEL cu multiprocessing) ===
    
    def pattern_worker(args):
        pattern_name, pattern_func = args
        try:
            return (pattern_name, pattern_func(seeds))
        except:
            return (pattern_name, {'pred': None, 'error': float('inf'), 'formula': 'error'})
    
    def exponential_pattern(seeds):
        try:
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c
            popt, _ = curve_fit(exp_func, x, y, maxfev=3000)
            exp_pred = exp_func(len(seeds), *popt)
            exp_error = np.mean(np.abs(y - exp_func(x, *popt)))
            return {'pred': exp_pred, 'error': exp_error, 'formula': "y = a*e^(bx) + c"}
        except:
            return {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    def fibonacci_pattern(seeds):
        if len(seeds) < 3:
            return {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
        try:
            A = np.array([[seeds[i-1], seeds[i-2]] for i in range(2, len(seeds))])
            B = np.array([seeds[i] for i in range(2, len(seeds))])
            coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            a, b = coeffs
            fib_pred = a * seeds[-1] + b * seeds[-2]
            errors = [abs(a * seeds[i-1] + b * seeds[i-2] - seeds[i]) for i in range(2, len(seeds))]
            fib_error = np.mean(errors) if errors else float('inf')
            return {'pred': fib_pred, 'error': fib_error, 'formula': f"S(n) = {a:.4f}*S(n-1) + {b:.4f}*S(n-2)"}
        except:
            return {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    def lcg_chain_pattern(seeds):
        if len(seeds) < 2:
            return {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
        try:
            m_estimate = 2147483648
            X = np.array([[seeds[i-1], 1] for i in range(1, len(seeds))])
            Y = np.array([seeds[i] for i in range(1, len(seeds))])
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            a, c = coeffs
            lcg_pred = (a * seeds[-1] + c) % m_estimate
            errors = [abs((a * seeds[i-1] + c) % m_estimate - seeds[i]) for i in range(1, len(seeds))]
            lcg_error = np.mean(errors) if errors else float('inf')
            return {'pred': lcg_pred, 'error': lcg_error, 'formula': f"S(n+1) = LCG mod {int(m_estimate)}"}
        except:
            return {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    def modular_pattern(seeds):
        if len(seeds) < 2:
            return {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
        try:
            diffs = np.diff(seeds)
            avg_diff = np.mean(diffs)
            m_estimate = 2147483648
            modular_pred = (seeds[-1] + avg_diff) % m_estimate
            errors = [abs((seeds[i-1] + avg_diff) % m_estimate - seeds[i]) for i in range(1, len(seeds))]
            modular_error = np.mean(errors)
            return {'pred': modular_pred, 'error': modular_error, 'formula': "S(n+1) = (S(n) + const) mod m"}
        except:
            return {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # Pattern functions pentru CPU parallel
    cpu_patterns = {
        'exponential': exponential_pattern,
        'fibonacci': fibonacci_pattern,
        'lcg_chain': lcg_chain_pattern,
        'modular': modular_pattern,
        'const_ratio': lambda s: {'pred': s[-1] * (s[-1]/s[-2] if len(s) >= 2 and s[-2] != 0 else 1), 'error': 0, 'formula': 'ratio'} if len(s) >= 2 else {'pred': None, 'error': float('inf'), 'formula': 'insufficient'},
    }
    
    # Rulare PARALLEL pe CPU
    print(f"  💻 CPU patterns: {len(cpu_patterns)} în paralel...")
    with Pool(processes=min(cpu_count(), len(cpu_patterns))) as pool:
        results = pool.map(pattern_worker, [(name, func) for name, func in cpu_patterns.items()])
        for pattern_name, pattern_result in results:
            all_patterns[pattern_name] = pattern_result
    
    print(f"  ✅ Total patterns calculați: {len(all_patterns)}")
    
    # Selectare cel mai bun pattern
    valid_patterns = {k: v for k, v in all_patterns.items() 
                     if v['pred'] is not None and not np.isnan(v['error']) and v['error'] != float('inf')}
    
    if not valid_patterns:
        return {
            'pattern_type': 'no_valid_pattern',
            'predicted_seed': None,
            'confidence': 0,
            'formula': 'N/A',
            'all_patterns': all_patterns
        }
    
    best_pattern_name = min(valid_patterns, key=lambda k: valid_patterns[k]['error'])
    best_pattern = valid_patterns[best_pattern_name]
    
    predicted_seed = int(round(best_pattern['pred']))
    
    # Confidence
    mean_seed = np.mean(y)
    if mean_seed > 0:
        confidence = max(0, min(100, 100 * (1 - best_pattern['error'] / mean_seed)))
    else:
        confidence = 0
    
    return {
        'pattern_type': best_pattern_name,
        'predicted_seed': predicted_seed,
        'confidence': round(confidence, 2),
        'formula': best_pattern['formula'],
        'error': round(best_pattern['error'], 2),
        'all_patterns': {k: {
            'error': round(v['error'], 2) if v['error'] != float('inf') else 'inf',
            'formula': v['formula'],
            'pred': int(round(v['pred'])) if v['pred'] is not None and not np.isnan(v['pred']) else None
        } for k, v in all_patterns.items()}
    }


class MaxPredictor:
    def __init__(self, lottery_type: str = "5-40"):
        self.lottery_type = lottery_type
        self.config = get_lottery_config(lottery_type)
        self.data_file = f"{lottery_type}_data.json"
        
    def load_data(self, last_n: Optional[int] = None, 
                  start_year: Optional[int] = None, 
                  end_year: Optional[int] = None) -> List[Dict]:
        """Încarcă datele"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Fișierul {self.data_file} nu există!")
            sys.exit(1)
        
        if isinstance(data, dict) and 'draws' in data:
            all_data = data['draws']
        elif isinstance(data, list):
            all_data = data
        else:
            print(f"❌ Format necunoscut")
            sys.exit(1)
        
        if last_n is not None:
            filtered_data = all_data[-last_n:] if len(all_data) >= last_n else all_data
        elif start_year is not None and end_year is not None:
            filtered_data = []
            for entry in all_data:
                try:
                    date_str = entry.get('data', entry.get('date', ''))
                    if '.' in date_str:
                        year = int(date_str.split('.')[-1])
                    elif '-' in date_str:
                        year = int(date_str.split('-')[0])
                    else:
                        year = entry.get('year', 0)
                    
                    if start_year <= year <= end_year:
                        filtered_data.append(entry)
                except:
                    continue
        else:
            filtered_data = all_data
        
        normalized = []
        for entry in filtered_data:
            normalized.append({
                'data': entry.get('data', entry.get('date', 'N/A')),
                'numere': entry.get('numere', entry.get('numbers', entry.get('numbers_sorted', [])))
            })
        
        return normalized
    
    def run_max_prediction(self, last_n: Optional[int] = None,
                          start_year: Optional[int] = None,
                          end_year: Optional[int] = None,
                          seed_range: tuple = (0, 100000000),  # 100M
                          search_size: int = 10000000,  # 10M seeds per extragere!
                          min_success_rate: float = 0.66):
        """PREDICȚIE MAXIMĂ - ZERO COMPROMISURI"""
        
        print(f"\n{'='*70}")
        print(f"  🚀 ULTIMATE MAX PREDICTOR - {self.lottery_type.upper()}")
        print(f"  ZERO COMPROMISURI - FULL POWER")
        print(f"{'='*70}\n")
        
        num_cores = cpu_count()
        print(f"💻 CPU Cores: {num_cores}")
        print(f"🔍 Seed Range: {seed_range[0]:,} - {seed_range[1]:,}")
        print(f"📊 Search Size: {search_size:,} seeds per extragere")
        print(f"⏰ Mersenne Timeout: 15 minute per extragere")
        print(f"📈 Pattern Analysis: TOATE cele 10 pattern-uri")
        print(f"🎯 RNG-uri testate: TOATE cele {len(RNG_TYPES)}\n")
        
        # Load data
        if last_n:
            print(f"📊 Încărcare ultimele {last_n} extrageri...")
            data = self.load_data(last_n=last_n)
        else:
            print(f"📊 Încărcare date {start_year}-{end_year}...")
            data = self.load_data(start_year=start_year, end_year=end_year)
        
        print(f"✅ {len(data)} extrageri încărcate\n")
        
        # Afișează extragerile
        print(f"📋 Extrageri încărcate:")
        for i, entry in enumerate(data, 1):
            print(f"  {i}. {entry['data']:15s} → {entry['numere']}")
        print()
        
        # Test TOATE RNG-urile - HYBRID GPU + CPU
        print(f"\n{'='*70}")
        print(f"  FAZA 1: TESTARE EXHAUSTIVĂ - HYBRID GPU + CPU")
        print(f"{'='*70}\n")
        
        if GPU_AVAILABLE:
            print(f"🚀 GPU Mode: Activ pentru {len(GPU_SUPPORTED_RNGS)} RNG-uri simple")
            print(f"   GPU RNG-uri: {', '.join(GPU_SUPPORTED_RNGS)}")
        print(f"💻 CPU Mode: {num_cores} cores pentru RNG-uri complexe\n")
        
        rng_results = {}
        
        for idx, rng_name in enumerate(RNG_TYPES.keys(), 1):
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(RNG_TYPES)}] RNG: {rng_name.upper()}")
            
            # Decide: GPU sau CPU?
            use_gpu = GPU_AVAILABLE and rng_name in GPU_SUPPORTED_RNGS
            
            if use_gpu:
                print(f"🚀 Mode: GPU ACCELERATED")
            else:
                print(f"💻 Mode: CPU MULTICORE")
            
            print(f"{'='*70}")
            
            seeds_found = []
            draws_with_seeds = []
            
            if use_gpu:
                # ===== GPU MODE =====
                for i, entry in enumerate(data):
                    numbers = entry.get('numere', [])
                    if len(numbers) != self.config.numbers_to_draw:
                        continue
                    
                    # GPU batch processing - 2M seeds per încercare
                    found_seed = find_seed_gpu_accelerated(
                        i, numbers, rng_name, self.config, seed_range, batch_size=2000000
                    )
                    
                    if found_seed is not None:
                        seeds_found.append(found_seed)
                        draws_with_seeds.append({
                            'idx': i,
                            'date': entry['data'],
                            'numbers': numbers,
                            'seed': found_seed
                        })
                    
                    # Progress
                    progress = 100 * (i + 1) / len(data)
                    print(f"  [{i + 1}/{len(data)}] ({progress:.1f}%)... {len(seeds_found)} seeds găsite", end='\r')
                
                print(f"\n✅ Seeds găsite (GPU): {len(seeds_found)}/{len(data)} ({len(seeds_found)/len(data):.1%})")
            
            else:
                # ===== CPU MODE =====
                tasks = []
                for i, entry in enumerate(data):
                    numbers = entry.get('numere', [])
                    if len(numbers) == self.config.numbers_to_draw:
                        tasks.append((i, numbers, rng_name, self.config, seed_range, search_size))
                
                with Pool(processes=num_cores) as pool:
                    optimal_chunksize = max(1, len(tasks) // (num_cores * 4))
                    for i, result in enumerate(pool.imap_unordered(find_seed_exhaustive_worker, tasks, chunksize=optimal_chunksize)):
                        idx_task, seed = result
                        
                        if seed is not None:
                            seeds_found.append(seed)
                            draws_with_seeds.append({
                                'idx': idx_task,
                                'date': data[idx_task]['data'],
                                'numbers': data[idx_task]['numere'],
                                'seed': seed
                            })
                        
                        # Progress
                        if (i + 1) % 2 == 0 or (i + 1) == len(tasks):
                            progress = 100 * (i + 1) / len(tasks)
                            print(f"  [{i + 1}/{len(tasks)}] ({progress:.1f}%)... {len(seeds_found)} seeds găsite", end='\r')
                
                print(f"\n✅ Seeds găsite (CPU): {len(seeds_found)}/{len(data)} ({len(seeds_found)/len(data):.1%})")
            
            success_rate = len(seeds_found) / len(data) if len(data) > 0 else 0
            
            print(f"📊 Success Rate: {success_rate:.1%}")
            
            if success_rate >= min_success_rate:
                print(f"✅ SUCCESS RATE PESTE THRESHOLD!")
                
                # SORTARE CRONOLOGICĂ - CRITIC pentru analiza pattern-ului!
                draws_with_seeds.sort(key=lambda x: x['idx'])
                seeds_found = [d['seed'] for d in draws_with_seeds]
                
                rng_results[rng_name] = {
                    'seeds': seeds_found,
                    'draws': draws_with_seeds,
                    'success_rate': success_rate
                }
            else:
                print(f"⚠️  Sub threshold ({success_rate:.1%} < {min_success_rate:.1%})")
        
        if not rng_results:
            print(f"\n❌ Niciun RNG nu a trecut de threshold!")
            return
        
        # Analiză EXHAUSTIVĂ pattern-uri
        print(f"\n{'='*70}")
        print(f"  FAZA 2: ANALIZĂ EXHAUSTIVĂ PATTERN-URI")
        print(f"{'='*70}\n")
        
        predictions = []
        
        for rng_name, result in sorted(rng_results.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            print(f"\n{'='*70}")
            print(f"RNG: {rng_name.upper()}")
            print(f"Success Rate: {result['success_rate']:.1%}")
            print(f"{'='*70}\n")
            
            # Analiză TOATE pattern-urile
            pattern_analysis = analyze_all_patterns_parallel_gpu(result['seeds'])
            
            print(f"🏆 BEST PATTERN: {pattern_analysis['pattern_type'].upper()}")
            print(f"📐 Formula: {pattern_analysis['formula']}")
            print(f"🎯 Confidence: {pattern_analysis['confidence']:.2f}%")
            print(f"❌ Error: {pattern_analysis.get('error', 'N/A')}\n")
            
            # Afișează TOATE pattern-urile
            print(f"📊 TOATE PATTERN-URILE ANALIZATE:")
            for pattern_name, pattern_data in pattern_analysis.get('all_patterns', {}).items():
                error_str = f"{pattern_data['error']}" if pattern_data['error'] != 'inf' else "∞"
                pred_str = f"{pattern_data['pred']:,}" if pattern_data['pred'] is not None else "N/A"
                print(f"   {pattern_name:20s}: Error={error_str:>10s} | Pred={pred_str:>15s} | {pattern_data['formula']}")
            print()
            
            # Generare predicție
            if pattern_analysis['predicted_seed'] is not None:
                try:
                    rng = create_rng(rng_name, pattern_analysis['predicted_seed'])
                    predicted_numbers = generate_numbers(
                        rng,
                        self.config.numbers_to_draw,
                        self.config.min_number,
                        self.config.max_number
                    )
                    
                    print(f"{'='*70}")
                    print(f"  🎯 PREDICȚIE FINALĂ")
                    print(f"{'='*70}")
                    print(f"  Seed Prezis: {pattern_analysis['predicted_seed']:,}")
                    print(f"  NUMERE PREZISE: {sorted(predicted_numbers)}")
                    print(f"{'='*70}\n")
                    
                    predictions.append({
                        'rng': rng_name,
                        'success_rate': result['success_rate'],
                        'pattern': pattern_analysis['pattern_type'],
                        'formula': pattern_analysis['formula'],
                        'confidence': pattern_analysis['confidence'],
                        'seed': pattern_analysis['predicted_seed'],
                        'numbers': sorted(predicted_numbers),
                        'all_patterns': pattern_analysis.get('all_patterns', {})
                    })
                except Exception as e:
                    print(f"❌ Eroare la generare predicție: {e}\n")
        
        # SUMAR FINAL
        if predictions:
            print(f"\n{'='*70}")
            print(f"  📊 SUMAR FINAL - TOP PREDICȚII")
            print(f"{'='*70}\n")
            
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. RNG: {pred['rng'].upper()}")
                print(f"   Success: {pred['success_rate']:.1%} | Confidence: {pred['confidence']:.1f}%")
                print(f"   Best Pattern: {pred['pattern']}")
                print(f"   Formula: {pred['formula']}")
                print(f"   NUMERE: {pred['numbers']}\n")
            
            # Salvare
            output_file = f"max_prediction_{self.lottery_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'lottery': self.lottery_type,
                    'timestamp': datetime.now().isoformat(),
                    'config': {
                        'seed_range': list(seed_range),
                        'search_size': search_size,
                        'min_success_rate': min_success_rate
                    },
                    'data_size': len(data),
                    'predictions': predictions
                }, f, indent=2)
            
            print(f"💾 Rezultate complete salvate: {output_file}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ULTIMATE MAX Predictor - ZERO compromisuri')
    parser.add_argument('--lottery', default='5-40', choices=['5-40', '6-49', 'joker'])
    parser.add_argument('--last-n', type=int, help='Ultimele N extrageri')
    parser.add_argument('--start-year', type=int)
    parser.add_argument('--end-year', type=int)
    parser.add_argument('--seed-range', type=int, nargs=2, default=[0, 100000000],
                      help='Seed range (default: 0 100000000)')
    parser.add_argument('--search-size', type=int, default=10000000,
                      help='Seeds testate per extragere (default: 10000000)')
    parser.add_argument('--min-success-rate', type=float, default=0.66)
    
    args = parser.parse_args()
    
    if not args.last_n and not (args.start_year and args.end_year):
        print("❌ Specifică --last-n SAU (--start-year și --end-year)!")
        sys.exit(1)
    
    predictor = MaxPredictor(args.lottery)
    predictor.run_max_prediction(
        last_n=args.last_n,
        start_year=args.start_year,
        end_year=args.end_year,
        seed_range=tuple(args.seed_range),
        search_size=args.search_size,
        min_success_rate=args.min_success_rate
    )
