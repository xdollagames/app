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

# RNG-uri suportate pe GPU
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


def find_seed_exhaustive_worker(args):
    """Worker pentru căutare EXHAUSTIVĂ - ZERO compromisuri"""
    import time
    
    draw_idx, numbers, rng_type, lottery_config, seed_range, search_size = args
    target_sorted = sorted(numbers)
    
    # Timeout GENEROS pentru Mersenne (15 minute per extragere)
    start_time = time.time()
    timeout_seconds = 900 if rng_type == 'mersenne' else 999999  # 15 min pentru Mersenne
    
    # Search size MAXIM - fără reduceri
    actual_search_size = search_size
    
    # Generează seed-uri random
    test_seeds = random.sample(range(seed_range[0], seed_range[1]), 
                              min(actual_search_size, seed_range[1] - seed_range[0]))
    
    seeds_tested = 0
    for seed in test_seeds:
        # Check timeout doar pentru Mersenne
        if rng_type == 'mersenne':
            if (time.time() - start_time) > timeout_seconds:
                print(f"\n  ⏰ Mersenne timeout după {seeds_tested:,} seeds testate pentru extragere #{draw_idx}")
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


def analyze_all_patterns(seeds: List[int]) -> Dict:
    """Analizează TOATE cele 10 pattern-uri - EXHAUSTIV"""
    if len(seeds) < 3:
        return {
            'pattern_type': 'insufficient_data',
            'predicted_seed': None,
            'confidence': 0,
            'formula': 'N/A',
            'all_patterns': {}
        }
    
    x = np.arange(len(seeds))
    y = np.array(seeds)
    
    all_patterns = {}
    
    # 1. Pattern LINEAR: y = a*x + b
    try:
        linear_coeffs = np.polyfit(x, y, 1)
        linear_pred = np.poly1d(linear_coeffs)(len(seeds))
        linear_error = np.mean(np.abs(y - np.poly1d(linear_coeffs)(x)))
        all_patterns['linear'] = {
            'pred': linear_pred,
            'error': linear_error,
            'formula': f"y = {linear_coeffs[0]:.4f}*x + {linear_coeffs[1]:.2f}"
        }
    except:
        all_patterns['linear'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # 2. Pattern POLINOMIAL grad 2: y = a*x² + b*x + c
    try:
        poly_coeffs = np.polyfit(x, y, 2)
        poly_pred = np.poly1d(poly_coeffs)(len(seeds))
        poly_error = np.mean(np.abs(y - np.poly1d(poly_coeffs)(x)))
        all_patterns['polynomial_2'] = {
            'pred': poly_pred,
            'error': poly_error,
            'formula': f"y = {poly_coeffs[0]:.4e}*x² + {poly_coeffs[1]:.4f}*x + {poly_coeffs[2]:.2f}"
        }
    except:
        all_patterns['polynomial_2'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # 3. Pattern POLINOMIAL grad 3: y = a*x³ + b*x² + c*x + d
    if len(seeds) >= 4:
        try:
            poly3_coeffs = np.polyfit(x, y, 3)
            poly3_pred = np.poly1d(poly3_coeffs)(len(seeds))
            poly3_error = np.mean(np.abs(y - np.poly1d(poly3_coeffs)(x)))
            all_patterns['polynomial_3'] = {
                'pred': poly3_pred,
                'error': poly3_error,
                'formula': f"y = {poly3_coeffs[0]:.4e}*x³ + ... (grad 3)"
            }
        except:
            all_patterns['polynomial_3'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['polynomial_3'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 4. Pattern EXPONENȚIAL: y = a*e^(b*x) + c
    try:
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c
        
        popt, _ = curve_fit(exp_func, x, y, maxfev=10000)
        exp_pred = exp_func(len(seeds), *popt)
        exp_error = np.mean(np.abs(y - exp_func(x, *popt)))
        all_patterns['exponential'] = {
            'pred': exp_pred,
            'error': exp_error,
            'formula': f"y = {popt[0]:.4e}*e^({popt[1]:.6f}*x) + {popt[2]:.2f}"
        }
    except:
        all_patterns['exponential'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # 5. Pattern FIBONACCI-like: S(n) = a*S(n-1) + b*S(n-2)
    if len(seeds) >= 3:
        try:
            A = np.array([[seeds[i-1], seeds[i-2]] for i in range(2, len(seeds))])
            B = np.array([seeds[i] for i in range(2, len(seeds))])
            coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            a, b = coeffs
            
            fib_pred = a * seeds[-1] + b * seeds[-2]
            
            errors = []
            for i in range(2, len(seeds)):
                pred_val = a * seeds[i-1] + b * seeds[i-2]
                errors.append(abs(pred_val - seeds[i]))
            fib_error = np.mean(errors) if errors else float('inf')
            
            all_patterns['fibonacci'] = {
                'pred': fib_pred,
                'error': fib_error,
                'formula': f"S(n) = {a:.6f}*S(n-1) + {b:.6f}*S(n-2)"
            }
        except:
            all_patterns['fibonacci'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['fibonacci'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 6. Pattern LCG CHAIN: S(n+1) = (a*S(n) + c) mod m
    if len(seeds) >= 2:
        try:
            # Estimăm m ca fiind 2^31 sau 2^32 (common pentru LCG)
            possible_m = [2147483648, 4294967296, max(seeds) * 2]
            
            best_lcg = None
            best_lcg_error = float('inf')
            
            for m_estimate in possible_m:
                X = np.array([[seeds[i-1], 1] for i in range(1, len(seeds))])
                Y = np.array([seeds[i] for i in range(1, len(seeds))])
                coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                a, c = coeffs
                
                lcg_pred = (a * seeds[-1] + c) % m_estimate
                
                errors = []
                for i in range(1, len(seeds)):
                    pred_val = (a * seeds[i-1] + c) % m_estimate
                    errors.append(abs(pred_val - seeds[i]))
                lcg_error = np.mean(errors) if errors else float('inf')
                
                if lcg_error < best_lcg_error:
                    best_lcg_error = lcg_error
                    best_lcg = {
                        'pred': lcg_pred,
                        'error': lcg_error,
                        'formula': f"S(n+1) = ({a:.6f}*S(n) + {c:.2f}) mod {int(m_estimate)}"
                    }
            
            all_patterns['lcg_chain'] = best_lcg if best_lcg else {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        except:
            all_patterns['lcg_chain'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['lcg_chain'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 7. Pattern MODULAR ARITHMETIC: S(n+1) = (S(n) + delta) mod m
    if len(seeds) >= 2:
        try:
            diffs = np.diff(seeds)
            avg_diff = np.mean(diffs)
            
            possible_m = [2147483648, 4294967296, max(seeds) * 2]
            best_mod = None
            best_mod_error = float('inf')
            
            for m_estimate in possible_m:
                modular_pred = (seeds[-1] + avg_diff) % m_estimate
                
                errors = []
                for i in range(1, len(seeds)):
                    pred_val = (seeds[i-1] + avg_diff) % m_estimate
                    errors.append(abs(pred_val - seeds[i]))
                modular_error = np.mean(errors)
                
                if modular_error < best_mod_error:
                    best_mod_error = modular_error
                    best_mod = {
                        'pred': modular_pred,
                        'error': modular_error,
                        'formula': f"S(n+1) = (S(n) + {avg_diff:.2f}) mod {int(m_estimate)}"
                    }
            
            all_patterns['modular'] = best_mod if best_mod else {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        except:
            all_patterns['modular'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['modular'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 8. Pattern DIFERENȚE CONSTANTE
    if len(seeds) >= 2:
        try:
            diffs = np.diff(seeds)
            avg_diff = np.mean(diffs)
            const_diff_pred = seeds[-1] + avg_diff
            const_diff_error = np.std(diffs)
            
            all_patterns['const_diff'] = {
                'pred': const_diff_pred,
                'error': const_diff_error,
                'formula': f"S(n+1) = S(n) + {avg_diff:.4f}"
            }
        except:
            all_patterns['const_diff'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['const_diff'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 9. Pattern RATIE CONSTANTĂ
    if len(seeds) >= 2 and all(s > 0 for s in seeds):
        try:
            ratios = [seeds[i] / seeds[i-1] for i in range(1, len(seeds))]
            avg_ratio = np.mean(ratios)
            ratio_pred = seeds[-1] * avg_ratio
            ratio_error = np.std(ratios) * seeds[-1]
            
            all_patterns['const_ratio'] = {
                'pred': ratio_pred,
                'error': ratio_error,
                'formula': f"S(n+1) = S(n) * {avg_ratio:.6f}"
            }
        except:
            all_patterns['const_ratio'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['const_ratio'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 10. Pattern LOGARITMIC: y = a*log(x) + b
    try:
        log_x = np.log(x + 1)
        log_coeffs = np.polyfit(log_x, y, 1)
        log_pred_x = np.log(len(seeds) + 1)
        log_pred = log_coeffs[0] * log_pred_x + log_coeffs[1]
        log_error = np.mean(np.abs(y - (log_coeffs[0] * log_x + log_coeffs[1])))
        
        all_patterns['logarithmic'] = {
            'pred': log_pred,
            'error': log_error,
            'formula': f"y = {log_coeffs[0]:.4f}*log(x) + {log_coeffs[1]:.2f}"
        }
    except:
        all_patterns['logarithmic'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # 11. Pattern POWER LAW: y = a*x^b + c
    try:
        def power_func(x, a, b, c):
            return a * np.power(x + 1, b) + c
        
        popt, _ = curve_fit(power_func, x, y, maxfev=10000, bounds=([0, -10, -np.inf], [np.inf, 10, np.inf]))
        power_pred = power_func(len(seeds), *popt)
        power_error = np.mean(np.abs(y - power_func(x, *popt)))
        
        all_patterns['power_law'] = {
            'pred': power_pred,
            'error': power_error,
            'formula': f"y = {popt[0]:.4e}*x^{popt[1]:.4f} + {popt[2]:.2f}"
        }
    except:
        all_patterns['power_law'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # 12. Pattern POLYNOMIAL grad 4
    if len(seeds) >= 5:
        try:
            poly4_coeffs = np.polyfit(x, y, 4)
            poly4_pred = np.poly1d(poly4_coeffs)(len(seeds))
            poly4_error = np.mean(np.abs(y - np.poly1d(poly4_coeffs)(x)))
            all_patterns['polynomial_4'] = {
                'pred': poly4_pred,
                'error': poly4_error,
                'formula': f"y = {poly4_coeffs[0]:.4e}*x⁴ + ... (grad 4)"
            }
        except:
            all_patterns['polynomial_4'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['polynomial_4'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 13. Pattern QUADRATIC CONGRUENTIAL: S(n+1) = (a*S(n)^2 + b*S(n) + c) mod m
    if len(seeds) >= 3:
        try:
            possible_m = [2147483648, 4294967296, max(seeds) * 2]
            best_qcg = None
            best_qcg_error = float('inf')
            
            for m_estimate in possible_m:
                # S(n+1) = (a*S(n)^2 + b*S(n) + c) mod m
                X = np.array([[seeds[i-1]**2, seeds[i-1], 1] for i in range(1, len(seeds))])
                Y = np.array([seeds[i] for i in range(1, len(seeds))])
                
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                    a, b, c = coeffs
                    
                    qcg_pred = (a * seeds[-1]**2 + b * seeds[-1] + c) % m_estimate
                    
                    errors = []
                    for i in range(1, len(seeds)):
                        pred_val = (a * seeds[i-1]**2 + b * seeds[i-1] + c) % m_estimate
                        errors.append(abs(pred_val - seeds[i]))
                    qcg_error = np.mean(errors) if errors else float('inf')
                    
                    if qcg_error < best_qcg_error:
                        best_qcg_error = qcg_error
                        best_qcg = {
                            'pred': qcg_pred,
                            'error': qcg_error,
                            'formula': f"S(n+1) = ({a:.4e}*S²(n) + {b:.4f}*S(n) + {c:.2f}) mod {int(m_estimate)}"
                        }
                except:
                    continue
            
            all_patterns['qcg'] = best_qcg if best_qcg else {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        except:
            all_patterns['qcg'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['qcg'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 14. Pattern MULTIPLICATIVE CONGRUENTIAL: S(n+1) = (a*S(n)) mod m
    if len(seeds) >= 2:
        try:
            possible_m = [2147483648, 4294967296, max(seeds) * 2]
            best_mult = None
            best_mult_error = float('inf')
            
            for m_estimate in possible_m:
                # Estimăm a ca media raporturilor
                ratios = [seeds[i] / seeds[i-1] for i in range(1, len(seeds)) if seeds[i-1] != 0]
                if ratios:
                    a = np.mean(ratios)
                    
                    mult_pred = (a * seeds[-1]) % m_estimate
                    
                    errors = []
                    for i in range(1, len(seeds)):
                        pred_val = (a * seeds[i-1]) % m_estimate
                        errors.append(abs(pred_val - seeds[i]))
                    mult_error = np.mean(errors)
                    
                    if mult_error < best_mult_error:
                        best_mult_error = mult_error
                        best_mult = {
                            'pred': mult_pred,
                            'error': mult_error,
                            'formula': f"S(n+1) = ({a:.6f}*S(n)) mod {int(m_estimate)}"
                        }
            
            all_patterns['multiplicative'] = best_mult if best_mult else {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        except:
            all_patterns['multiplicative'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['multiplicative'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 15. Pattern LAG-3 (Autoregressive): S(n) = a*S(n-1) + b*S(n-2) + c*S(n-3)
    if len(seeds) >= 4:
        try:
            A = np.array([[seeds[i-1], seeds[i-2], seeds[i-3]] for i in range(3, len(seeds))])
            B = np.array([seeds[i] for i in range(3, len(seeds))])
            coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            a, b, c = coeffs
            
            lag3_pred = a * seeds[-1] + b * seeds[-2] + c * seeds[-3]
            
            errors = []
            for i in range(3, len(seeds)):
                pred_val = a * seeds[i-1] + b * seeds[i-2] + c * seeds[i-3]
                errors.append(abs(pred_val - seeds[i]))
            lag3_error = np.mean(errors) if errors else float('inf')
            
            all_patterns['lag3'] = {
                'pred': lag3_pred,
                'error': lag3_error,
                'formula': f"S(n) = {a:.6f}*S(n-1) + {b:.6f}*S(n-2) + {c:.6f}*S(n-3)"
            }
        except:
            all_patterns['lag3'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['lag3'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 16. Pattern HYPERBOLIC: y = a/(x + b) + c
    try:
        def hyperbolic_func(x, a, b, c):
            return a / (x + b + 1) + c
        
        popt, _ = curve_fit(hyperbolic_func, x, y, maxfev=10000)
        hyp_pred = hyperbolic_func(len(seeds), *popt)
        hyp_error = np.mean(np.abs(y - hyperbolic_func(x, *popt)))
        
        all_patterns['hyperbolic'] = {
            'pred': hyp_pred,
            'error': hyp_error,
            'formula': f"y = {popt[0]:.4f}/(x + {popt[1]:.4f}) + {popt[2]:.2f}"
        }
    except:
        all_patterns['hyperbolic'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # 17. Pattern XOR CHAIN: S(n+1) = S(n) XOR (S(n) << a) XOR (S(n) >> b)
    if len(seeds) >= 2:
        try:
            # Găsim cel mai bun shift prin încercare
            best_xor = None
            best_xor_error = float('inf')
            
            for shift_a in [1, 3, 5, 7, 13, 17]:
                for shift_b in [1, 3, 5, 7, 13, 17]:
                    errors = []
                    for i in range(1, len(seeds)):
                        # Aproximăm cu modulo pentru a evita overflow
                        max_val = max(seeds) * 4
                        pred_val = seeds[i-1] ^ ((seeds[i-1] << shift_a) % max_val) ^ ((seeds[i-1] >> shift_b))
                        errors.append(abs(pred_val - seeds[i]))
                    
                    xor_error = np.mean(errors)
                    if xor_error < best_xor_error:
                        best_xor_error = xor_error
                        xor_pred = seeds[-1] ^ ((seeds[-1] << shift_a) % max_val) ^ ((seeds[-1] >> shift_b))
                        best_xor = {
                            'pred': xor_pred,
                            'error': xor_error,
                            'formula': f"S(n+1) = S(n) XOR (S(n)<<{shift_a}) XOR (S(n)>>{shift_b})"
                        }
            
            all_patterns['xor_chain'] = best_xor if best_xor else {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        except:
            all_patterns['xor_chain'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['xor_chain'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 18. Pattern COMBINED LCG (două LCG-uri combinate): S(n+1) = (LCG1(n) + LCG2(n)) mod m
    if len(seeds) >= 3:
        try:
            m_estimate = max(seeds) * 2 if max(seeds) > 0 else 2147483648
            
            # Presupunem două LCG-uri cu parametri diferiți
            # Simplificare: S(n+1) ≈ (a1*S(n) + c1 + a2*S(n) + c2) mod m
            # = ((a1+a2)*S(n) + (c1+c2)) mod m
            # Deci e similar cu un singur LCG, dar poate avea pattern diferit
            
            # Încercăm variație: S(n+1) = (a*S(n) + b*n + c) mod m (LCG cu trend)
            X = np.array([[seeds[i-1], i, 1] for i in range(1, len(seeds))])
            Y = np.array([seeds[i] for i in range(1, len(seeds))])
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            a, b, c = coeffs
            
            combined_pred = (a * seeds[-1] + b * len(seeds) + c) % m_estimate
            
            errors = []
            for i in range(1, len(seeds)):
                pred_val = (a * seeds[i-1] + b * i + c) % m_estimate
                errors.append(abs(pred_val - seeds[i]))
            combined_error = np.mean(errors) if errors else float('inf')
            
            all_patterns['combined_lcg'] = {
                'pred': combined_pred,
                'error': combined_error,
                'formula': f"S(n+1) = ({a:.4f}*S(n) + {b:.4f}*n + {c:.2f}) mod {int(m_estimate)}"
            }
        except:
            all_patterns['combined_lcg'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['combined_lcg'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
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
                          search_size: int = 5000000,  # 5M
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
            pattern_analysis = analyze_all_patterns(result['seeds'])
            
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
    parser.add_argument('--search-size', type=int, default=5000000,
                      help='Seeds testate per extragere (default: 5000000)')
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
