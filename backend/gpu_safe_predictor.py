#!/usr/bin/env python3
"""
GPU-SAFE MAX PREDICTOR
Respectă TOATE regulile CUDA + Multiprocessing
"""

import json
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
import random
import threading
from queue import Queue

# ❌ NU importăm CuPy aici!
# ❌ NU setăm fork!
# ✅ Workers nu vor atinge CUDA niciodată

from lottery_config import get_lottery_config
from advanced_rng_library import RNG_TYPES, create_rng, generate_numbers


def compute_modular_inverse(a, m):
    """Inversul modular"""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        return gcd, y1 - (b // a) * x1, x1
    
    gcd, x, _ = extended_gcd(a % m, m)
    return (x % m + m) % m if gcd == 1 else None


def reverse_lcg_generic(output, min_num, max_num, a, c, m):
    """Reverse LCG generic"""
    range_size = max_num - min_num + 1
    target_mod = output - min_num
    a_inv = compute_modular_inverse(a, m)
    
    if a_inv is None:
        return None
    
    for k in range(min(1000, m // range_size + 1)):
        state = target_mod + k * range_size
        if state >= m:
            break
        prev_state = (a_inv * (state - c)) % m
        if 0 <= prev_state < m:
            return prev_state
    return None


def try_reverse_engineering(rng_name, numbers, lottery_config):
    """Reverse engineering pentru LCG-uri"""
    if not numbers:
        return None
    
    reversible = {
        'lcg_glibc': (1103515245, 12345, 2**31),
        'lcg_minstd': (48271, 0, 2**31 - 1),
        'lcg_randu': (65539, 0, 2**31),
        'lcg_borland': (22695477, 1, 2**32),
        'lcg_weak': (9301, 49297, 233280),
        'php_rand': (1103515245, 12345, 0x7FFFFFFF),
    }
    
    if rng_name in reversible:
        a, c, m = reversible[rng_name]
        seed = reverse_lcg_generic(numbers[0], lottery_config.min_number, lottery_config.max_number, a, c, m)
        
        if seed is not None:
            try:
                rng = create_rng(rng_name, seed)
                generated = generate_numbers(rng, lottery_config.numbers_to_draw, lottery_config.min_number, lottery_config.max_number)
                if sorted(generated) == sorted(numbers):
                    return seed
            except:
                pass
    
    return None


def cpu_worker(args):
    """Worker CPU - NU atinge CUDA niciodată!"""
    draw_idx, numbers, rng_name, lottery_config, seed_range, search_size = args
    target_sorted = sorted(numbers)
    
    # Reverse engineering
    reversed_seed = try_reverse_engineering(rng_name, numbers, lottery_config)
    if reversed_seed is not None:
        return (draw_idx, reversed_seed)
    
    # Brute force
    test_seeds = random.sample(range(seed_range[0], seed_range[1]), min(search_size, seed_range[1] - seed_range[0]))
    
    for seed in test_seeds:
        try:
            rng = create_rng(rng_name, seed)
            generated = generate_numbers(rng, lottery_config.numbers_to_draw, lottery_config.min_number, lottery_config.max_number)
            if sorted(generated) == target_sorted:
                return (draw_idx, seed)
        except:
            continue
    
    return (draw_idx, None)


def gpu_thread_worker(data, lottery_config, seed_range, results_queue):
    """GPU Thread - testează 12 RNG-uri SECVENȚIAL cu TOT GPU-ul"""
    try:
        import cupy as cp
        print("🚀 [GPU Thread] CuPy importat cu succes!\n")
        
        # 1 RNG pentru GPU (doar xorshift_simple - singurul cu kernel CORECT!)
        gpu_rngs_to_test = ['xorshift_simple']
        
        print(f"🚀 [GPU] Va testa {len(gpu_rngs_to_test)} RNG (xorshift_simple - kernel CORECT)")
        print(f"⚠️  Alte RNG-uri necesită kernels dedicați - momentan pe CPU\n")
        
        # Kernel xorshift (funcționează pentru xorshift_simple, xorshift32, xorshift64, xorshift128)
        xorshift_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void test_xorshift(unsigned int* seeds, int num_seeds, int* target, int target_size, 
                          int min_num, int max_num, int* results) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_seeds) return;
            
            unsigned int state = seeds[idx];
            int range = max_num - min_num + 1;
            int gen[10];
            
            for (int i = 0; i < target_size; i++) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                gen[i] = min_num + (state % range);
            }
            
            // Sort
            for (int i = 0; i < target_size - 1; i++) {
                for (int j = 0; j < target_size - i - 1; j++) {
                    if (gen[j] > gen[j+1]) {
                        int temp = gen[j]; gen[j] = gen[j+1]; gen[j+1] = temp;
                    }
                }
            }
            
            // Compare
            int match = 1;
            for (int i = 0; i < target_size; i++) {
                if (gen[i] != target[i]) {
                    match = 0;
                    break;
                }
            }
            results[idx] = match;
        }
        ''', 'test_xorshift')
        
        gpu_results = {}
        
        # Test TOATE RNG-urile GPU (folosind kernel generic pentru demo)
        for rng_name in gpu_rngs_to_test:
            print(f"🚀 [GPU] Testing: {rng_name.upper()}")
            
            seeds_found = []
            draws_with_seeds = []
            
            for draw_idx, entry in enumerate(data):
                numbers = entry.get('numere', [])
                if len(numbers) != lottery_config.numbers_to_draw:
                    continue
                
                target_sorted = sorted(numbers)
                found = False
                
                # GPU batch testing - folosește TOT GPU-ul
                batch_size = 5000000  # 5M seeds per batch (mai mult decât înainte!)
                max_batches = 20
                
                for batch_num in range(max_batches):
                    seeds_batch = cp.random.randint(0, seed_range[1], size=batch_size, dtype=cp.uint32)
                    target_gpu = cp.array(target_sorted, dtype=cp.int32)
                    results_gpu = cp.zeros(batch_size, dtype=cp.int32)
                    
                    threads = 512  # Mai multe threads
                    blocks = (batch_size + threads - 1) // threads
                    
                    xorshift_kernel((blocks,), (threads,), 
                               (seeds_batch, batch_size, target_gpu, len(target_sorted),
                                lottery_config.min_number, lottery_config.max_number, results_gpu))
                    
                    matches = cp.where(results_gpu == 1)[0]
                    
                    if len(matches) > 0:
                        seed_found = int(cp.asnumpy(seeds_batch[matches[0]]))
                        seeds_found.append(seed_found)
                        draws_with_seeds.append({
                            'idx': draw_idx,
                            'date': entry['data'],
                            'numbers': numbers,
                            'seed': seed_found
                        })
                        found = True
                        break
                
                progress = 100 * (draw_idx + 1) / len(data)
                print(f"  [{draw_idx+1}/{len(data)}] ({progress:.1f}%)... {len(seeds_found)} seeds găsite", end='\r')
            
            success_rate = len(seeds_found) / len(data) if len(data) > 0 else 0
            print(f"\n✅ [GPU] {rng_name}: {len(seeds_found)}/{len(data)} ({success_rate:.1%})")
            
            if success_rate >= 0.66:
                print(f"  ✅ Peste threshold 66%!")
                draws_with_seeds.sort(key=lambda x: x['idx'])
                seeds_found = [d['seed'] for d in draws_with_seeds]
                gpu_results[rng_name] = {
                    'seeds': seeds_found,
                    'draws': draws_with_seeds,
                    'success_rate': success_rate
                }
            else:
                print(f"  ❌ Sub threshold ({success_rate:.1%} < 66%)")
            
            print()
        
        results_queue.put(('gpu', gpu_results))
        
    except Exception as e:
        print(f"⚠️  [GPU Thread] Error: {e}")
        import traceback
        traceback.print_exc()
        results_queue.put(('gpu', {}))


def cpu_multiprocessing_worker(data, lottery_config, seed_range, search_size, min_success_rate, results_queue):
    """CPU Thread - testează 9 RNG-uri SECVENȚIAL cu (total_cores - 3)"""
    
    # Verificare cores ÎNAINTE de a porni
    total_cores = cpu_count()
    num_cores = max(1, total_cores - 3)  # Lasă 3 cores pentru GPU thread
    
    print(f"💻 [CPU Thread] Total cores: {total_cores}")
    print(f"💻 [CPU Thread] Folosește: {num_cores} cores (lasă 3 pentru GPU)")
    
    # RNG-uri CPU (exclude doar xorshift_simple testat pe GPU)
    gpu_rngs = ['xorshift_simple']  # Doar acesta are kernel corect!
    
    cpu_rngs = [r for r in RNG_TYPES.keys() if r not in gpu_rngs]
    
    print(f"💻 [CPU] Va testa {len(cpu_rngs)} RNG-uri SECVENȚIAL (fiecare cu {num_cores} cores)")
    print(f"   RNG-uri CPU: {', '.join(cpu_rngs[:5])}... (+{len(cpu_rngs)-5} altele)\n")
    
    cpu_results = {}
    
    for rng_name in cpu_rngs:
        print(f"💻 [CPU] Testing: {rng_name.upper()}")
        
        tasks = [(i, e['numere'], rng_name, lottery_config, seed_range, search_size) 
                for i, e in enumerate(data) if len(e['numere']) == lottery_config.numbers_to_draw]
        
        seeds_found = []
        draws_with_seeds = []
        
        # Folosește TOATE cores-urile pentru acest RNG
        with Pool(processes=num_cores) as pool:
            for i, result in enumerate(pool.imap_unordered(cpu_worker, tasks)):
                idx_task, seed = result
                if seed is not None:
                    seeds_found.append(seed)
                    draws_with_seeds.append({
                        'idx': idx_task,
                        'date': data[idx_task]['data'],
                        'numbers': data[idx_task]['numere'],
                        'seed': seed
                    })
                
                if (i + 1) % 2 == 0 or (i + 1) == len(tasks):
                    progress = 100 * (i + 1) / len(tasks)
                    print(f"  [{i+1}/{len(tasks)}] ({progress:.1f}%)... {len(seeds_found)} seeds găsite", end='\r')
        
        success_rate = len(seeds_found) / len(data) if len(data) > 0 else 0
        print(f"\n✅ [CPU] {rng_name}: {len(seeds_found)}/{len(data)} ({success_rate:.1%})")
        
        if success_rate >= min_success_rate:
            print(f"  ✅ Peste threshold {min_success_rate:.1%}!")
            draws_with_seeds.sort(key=lambda x: x['idx'])
            seeds_found = [d['seed'] for d in draws_with_seeds]
            cpu_results[rng_name] = {
                'seeds': seeds_found,
                'draws': draws_with_seeds,
                'success_rate': success_rate
            }
        else:
            print(f"  ❌ Sub threshold ({success_rate:.1%} < {min_success_rate:.1%})")
        
        print()
    
    results_queue.put(('cpu', cpu_results))


def analyze_patterns_parallel_gpu_cpu(seeds):
    """Pattern analysis - GPU + CPU PARALEL cu TOATE 23 pattern-urile!"""
    if len(seeds) < 3:
        return {'pattern_type': 'insufficient', 'predicted_seed': None, 'confidence': 0}
    
    print(f"  🎯 Analiză PARALEL: GPU patterns + CPU patterns SIMULTAN...")
    
    x = np.arange(len(seeds))
    y = np.array(seeds)
    
    all_patterns = {}
    pattern_queue = Queue()
    
    # === GPU THREAD pentru pattern-uri simple ===
    def gpu_patterns_thread():
        gpu_patt = {}
        try:
            import cupy as cp
            x_gpu = cp.asarray(x, dtype=cp.float64)
            y_gpu = cp.asarray(y, dtype=cp.float64)
            
            # 1-4: Polynomial (LINEAR, POLY2, POLY3, POLY4)
            for deg in [1, 2, 3, 4]:
                if len(seeds) >= deg + 1:
                    try:
                        coeffs = cp.polyfit(x_gpu, y_gpu, deg)
                        pred = float(cp.asnumpy(cp.poly1d(coeffs)(len(seeds))))
                        error = float(cp.asnumpy(cp.mean(cp.abs(y_gpu - cp.poly1d(coeffs)(x_gpu)))))
                        name = 'linear' if deg == 1 else f'poly_{deg}'
                        gpu_patt[name] = {'pred': pred, 'error': error, 'formula': f'poly(deg={deg})'}
                    except:
                        pass
            
            # 5. Logarithmic
            if len(seeds) >= 2:
                try:
                    log_x = cp.log(x_gpu + 1)
                    log_c = cp.polyfit(log_x, y_gpu, 1)
                    pred = float(cp.asnumpy(log_c[0] * cp.log(len(seeds) + 1) + log_c[1]))
                    error = float(cp.asnumpy(cp.mean(cp.abs(y_gpu - (log_c[0] * log_x + log_c[1])))))
                    gpu_patt['logarithmic'] = {'pred': pred, 'error': error, 'formula': 'log(x)'}
                except:
                    pass
            
            # 6. Const Diff
            if len(seeds) >= 2:
                try:
                    diffs = cp.diff(y_gpu)
                    avg_diff = cp.mean(diffs)
                    pred = float(y_gpu[-1] + avg_diff)
                    error = float(cp.std(diffs))
                    gpu_patt['const_diff'] = {'pred': pred, 'error': error, 'formula': 'S(n+1)=S(n)+const'}
                except:
                    pass
            
            # 7. Const Ratio
            if len(seeds) >= 2 and all(s > 0 for s in seeds):
                try:
                    ratios = y_gpu[1:] / y_gpu[:-1]
                    avg_ratio = cp.mean(ratios)
                    pred = float(y_gpu[-1] * avg_ratio)
                    error = float(cp.std(ratios) * y_gpu[-1])
                    gpu_patt['const_ratio'] = {'pred': pred, 'error': error, 'formula': 'S(n+1)=S(n)*ratio'}
                except:
                    pass
            
            print(f"  ✅ GPU: {len(gpu_patt)} patterns calculați")
        except:
            print(f"  ⚠️ GPU patterns: fallback CPU")
        
        pattern_queue.put(('gpu', gpu_patt))
    
    # === CPU THREAD pentru pattern-uri complexe (16 pattern-uri!) ===
    def cpu_patterns_thread():
        cpu_patt = {}
        
        # 8. Exponential
        try:
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c
            popt, _ = curve_fit(exp_func, x, y, maxfev=3000)
            pred = exp_func(len(seeds), *popt)
            error = np.mean(np.abs(y - exp_func(x, *popt)))
            cpu_patt['exponential'] = {'pred': pred, 'error': error, 'formula': 'exponential'}
        except:
            pass
        
        # 9. Fibonacci
        if len(seeds) >= 3:
            try:
                A = np.array([[seeds[i-1], seeds[i-2]] for i in range(2, len(seeds))])
                B = np.array([seeds[i] for i in range(2, len(seeds))])
                coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
                a, b = coeffs
                pred = a * seeds[-1] + b * seeds[-2]
                errors = [abs(a * seeds[i-1] + b * seeds[i-2] - seeds[i]) for i in range(2, len(seeds))]
                cpu_patt['fibonacci'] = {'pred': pred, 'error': np.mean(errors), 'formula': f'Fib: {a:.4f}*S(n-1)+{b:.4f}*S(n-2)'}
            except:
                pass
        
        # 10. LCG Chain (IMPORTANT!)
        if len(seeds) >= 2:
            try:
                m = 2147483648
                X = np.array([[seeds[i-1], 1] for i in range(1, len(seeds))])
                Y = np.array([seeds[i] for i in range(1, len(seeds))])
                coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                a, c = coeffs
                pred = (a * seeds[-1] + c) % m
                errors = [abs((a * seeds[i-1] + c) % m - seeds[i]) for i in range(1, len(seeds))]
                cpu_patt['lcg_chain'] = {'pred': pred, 'error': np.mean(errors), 'formula': f'S(n+1)=({a:.4f}*S(n)+{c:.2f})mod{m}'}
            except:
                pass
        
        # 11. Modular
        if len(seeds) >= 2:
            try:
                diffs = np.diff(seeds)
                avg_diff = np.mean(diffs)
                m = 2147483648
                pred = (seeds[-1] + avg_diff) % m
                errors = [abs((seeds[i-1] + avg_diff) % m - seeds[i]) for i in range(1, len(seeds))]
                cpu_patt['modular'] = {'pred': pred, 'error': np.mean(errors), 'formula': f'S(n+1)=(S(n)+{avg_diff:.2f})mod{m}'}
            except:
                pass
        
        # 12. Power Law
        try:
            def power_func(x, a, b, c):
                return a * np.power(x + 1, b) + c
            popt, _ = curve_fit(power_func, x, y, maxfev=3000, bounds=([0, -10, -np.inf], [np.inf, 10, np.inf]))
            pred = power_func(len(seeds), *popt)
            error = np.mean(np.abs(y - power_func(x, *popt)))
            cpu_patt['power_law'] = {'pred': pred, 'error': error, 'formula': 'power_law'}
        except:
            pass
        
        # 13. QCG (Quadratic Congruential)
        if len(seeds) >= 3:
            try:
                m = 2147483648
                X = np.array([[seeds[i-1]**2, seeds[i-1], 1] for i in range(1, len(seeds))])
                Y = np.array([seeds[i] for i in range(1, len(seeds))])
                coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                a, b, c = coeffs
                pred = (a * seeds[-1]**2 + b * seeds[-1] + c) % m
                errors = [abs((a * seeds[i-1]**2 + b * seeds[i-1] + c) % m - seeds[i]) for i in range(1, len(seeds))]
                cpu_patt['qcg'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'QCG'}
            except:
                pass
        
        # 14. Multiplicative
        if len(seeds) >= 2:
            try:
                m = 2147483648
                ratios = [seeds[i] / seeds[i-1] for i in range(1, len(seeds)) if seeds[i-1] != 0]
                if ratios:
                    a = np.mean(ratios)
                    pred = (a * seeds[-1]) % m
                    errors = [abs((a * seeds[i-1]) % m - seeds[i]) for i in range(1, len(seeds))]
                    cpu_patt['multiplicative'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'multiplicative'}
            except:
                pass
        
        # 15. Lag-3
        if len(seeds) >= 4:
            try:
                A = np.array([[seeds[i-1], seeds[i-2], seeds[i-3]] for i in range(3, len(seeds))])
                B = np.array([seeds[i] for i in range(3, len(seeds))])
                coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
                a, b, c = coeffs
                pred = a * seeds[-1] + b * seeds[-2] + c * seeds[-3]
                errors = [abs(a * seeds[i-1] + b * seeds[i-2] + c * seeds[i-3] - seeds[i]) for i in range(3, len(seeds))]
                cpu_patt['lag3'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'lag3'}
            except:
                pass
        
        # 16. Hyperbolic
        try:
            def hyp_func(x, a, b, c):
                return a / (x + b + 1) + c
            popt, _ = curve_fit(hyp_func, x, y, maxfev=3000)
            pred = hyp_func(len(seeds), *popt)
            error = np.mean(np.abs(y - hyp_func(x, *popt)))
            cpu_patt['hyperbolic'] = {'pred': pred, 'error': error, 'formula': 'hyperbolic'}
        except:
            pass
        
        # 17. XOR Chain
        if len(seeds) >= 2:
            try:
                max_val = max(seeds) * 4 if max(seeds) > 0 else 0xFFFFFFFF
                best_err = float('inf')
                best_pred = None
                for sa in [5, 7, 13]:
                    for sb in [5, 7, 13]:
                        errors = [abs((seeds[i-1] ^ ((seeds[i-1] << sa) % max_val) ^ (seeds[i-1] >> sb)) - seeds[i]) for i in range(1, len(seeds))]
                        err = np.mean(errors)
                        if err < best_err:
                            best_err = err
                            best_pred = seeds[-1] ^ ((seeds[-1] << sa) % max_val) ^ (seeds[-1] >> sb)
                cpu_patt['xor_chain'] = {'pred': best_pred, 'error': best_err, 'formula': 'xor_chain'}
            except:
                pass
        
        # 18. Combined LCG
        if len(seeds) >= 3:
            try:
                m = 2147483648
                X = np.array([[seeds[i-1], i, 1] for i in range(1, len(seeds))])
                Y = np.array([seeds[i] for i in range(1, len(seeds))])
                coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                a, b, c = coeffs
                pred = (a * seeds[-1] + b * len(seeds) + c) % m
                errors = [abs((a * seeds[i-1] + b * i + c) % m - seeds[i]) for i in range(1, len(seeds))]
                cpu_patt['combined_lcg'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'combined_lcg'}
            except:
                pass
        
        # 19. Hash Mix
        if len(seeds) >= 2:
            try:
                c1, r1, c2 = 0xcc9e2d51, 15, 0x1b873593
                errors = [abs((((seeds[i-1] * c1) & 0xFFFFFFFF) ^ (seeds[i-1] >> r1)) * c2 & 0xFFFFFFFF - seeds[i]) for i in range(1, len(seeds))]
                pred = (((seeds[-1] * c1) & 0xFFFFFFFF) ^ (seeds[-1] >> r1)) * c2 & 0xFFFFFFFF
                cpu_patt['hash_mix'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'hash_mix'}
            except:
                pass
        
        # 20. Hash Rotate
        if len(seeds) >= 2:
            try:
                k, c = 13, 0x9e3779b9
                errors = [abs((((seeds[i-1] << k) | (seeds[i-1] >> (32 - k))) & 0xFFFFFFFF) ^ c - seeds[i]) for i in range(1, len(seeds))]
                pred = (((seeds[-1] << k) | (seeds[-1] >> (32 - k))) & 0xFFFFFFFF) ^ c
                cpu_patt['hash_rotate'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'hash_rotate'}
            except:
                pass
        
        # 21. Hash Weyl
        if len(seeds) >= 2:
            try:
                weyl = 0x9e3779b97f4a7c15
                pred = (seeds[-1] + weyl) & 0xFFFFFFFF
                errors = [abs((seeds[i-1] + weyl) & 0xFFFFFFFF - seeds[i]) for i in range(1, len(seeds))]
                cpu_patt['hash_weyl'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'hash_weyl'}
            except:
                pass
        
        # 22. Hash Combine (SplitMix)
        if len(seeds) >= 2:
            try:
                gamma = 0x9e3779b97f4a7c15
                z = (seeds[-1] + gamma) & 0xFFFFFFFFFFFFFFFF
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
                z = (z ^ (z >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
                pred = (z ^ (z >> 31)) & 0xFFFFFFFF
                errors = []
                for i in range(1, len(seeds)):
                    z = (seeds[i-1] + gamma) & 0xFFFFFFFFFFFFFFFF
                    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
                    z = (z ^ (z >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
                    p = (z ^ (z >> 31)) & 0xFFFFFFFF
                    errors.append(abs(p - seeds[i]))
                cpu_patt['hash_combine'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'hash_combine'}
            except:
                pass
        
        # 23. Hash Avalanche
        if len(seeds) >= 2:
            try:
                def avalanche(x):
                    x = (x ^ (x >> 30)) & 0xFFFFFFFFFFFFFFFF
                    x = (x * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
                    x = (x ^ (x >> 27)) & 0xFFFFFFFFFFFFFFFF
                    x = (x * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
                    return (x ^ (x >> 31)) & 0xFFFFFFFF
                
                errors = [abs(avalanche(seeds[i-1]) - seeds[i]) for i in range(1, len(seeds))]
                pred = avalanche(seeds[-1])
                cpu_patt['hash_avalanche'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'hash_avalanche'}
            except:
                pass
        
        print(f"  ✅ CPU: {len(cpu_patt)} patterns calculați")
        pattern_queue.put(('cpu', cpu_patt))
    
    # Launch BOTH threads SIMULTAN
    gpu_t = threading.Thread(target=gpu_patterns_thread)
    cpu_t = threading.Thread(target=cpu_patterns_thread)
    
    gpu_t.start()
    cpu_t.start()
    
    gpu_t.join()
    cpu_t.join()
    
    # Colectează
    while not pattern_queue.empty():
        source, patterns = pattern_queue.get()
        all_patterns.update(patterns)
    
    print(f"  ✅ Total: {len(all_patterns)} patterns analizați\n")
    
    # Găsește pattern-uri valide
    valid = {k: v for k, v in all_patterns.items() if v.get('pred') and v.get('error') != float('inf')}
    
    if not valid:
        return {'pattern_type': 'none', 'predicted_seed': None, 'confidence': 0, 'all_patterns': all_patterns, 'top_patterns': []}
    
    # Calculează confidence pentru fiecare
    patterns_with_confidence = []
    mean_y = np.mean(y)
    
    for name, patt in valid.items():
        conf = max(0, min(100, 100 * (1 - patt['error'] / mean_y))) if mean_y > 0 else 0
        patterns_with_confidence.append({
            'name': name,
            'pred': int(round(patt['pred'])),
            'error': round(patt['error'], 2),
            'confidence': round(conf, 2),
            'formula': patt.get('formula', name)
        })
    
    # Sortează după confidence
    patterns_with_confidence.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Verifică dacă există pattern-uri cu 100% confidence
    perfect_patterns = [p for p in patterns_with_confidence if p['confidence'] == 100.0]
    
    if perfect_patterns:
        # RETURNEAZĂ TOATE cu 100%
        return {
            'pattern_type': 'multiple_perfect' if len(perfect_patterns) > 1 else perfect_patterns[0]['name'],
            'predicted_seed': perfect_patterns[0]['pred'],  # Primul seed (toate ar trebui să fie identice dacă sunt perfect)
            'confidence': 100.0,
            'formula': perfect_patterns[0]['formula'],
            'error': 0.0,
            'all_patterns': all_patterns,
            'top_patterns': perfect_patterns  # TOATE cu 100%
        }
    else:
        # Returnează doar cel mai bun
        best = patterns_with_confidence[0]
        return {
            'pattern_type': best['name'],
            'predicted_seed': best['pred'],
            'confidence': best['confidence'],
            'formula': best['formula'],
            'error': best['error'],
            'all_patterns': all_patterns,
            'top_patterns': [best]  # Doar cel mai bun
        }


class GPUSafePredictor:
    def __init__(self, lottery_type="5-40"):
        self.lottery_type = lottery_type
        self.config = get_lottery_config(lottery_type)
        self.data_file = f"{lottery_type}_data.json"
    
    def load_data(self, last_n=None, start_year=None, end_year=None):
        """Load data"""
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        all_data = data['draws'] if 'draws' in data else data
        
        if last_n:
            filtered = all_data[-last_n:]
        elif start_year and end_year:
            filtered = []
            for e in all_data:
                try:
                    date_str = e.get('data', e.get('date', ''))
                    year = int(date_str.split('.')[-1] if '.' in date_str else date_str.split('-')[0])
                    if start_year <= year <= end_year:
                        filtered.append(e)
                except:
                    continue
        else:
            filtered = all_data
        
        return [{'data': e.get('data', e.get('date')), 'numere': e.get('numere', e.get('numbers', e.get('numbers_sorted')))} for e in filtered]
    
    def run_prediction(self, last_n=None, start_year=None, end_year=None,
                      seed_range=(0, 100000000), search_size=10000000, min_success_rate=0.66):
        
        print(f"\n{'='*70}")
        print(f"  GPU-SAFE PREDICTOR - {self.lottery_type.upper()}")
        print(f"{'='*70}\n")
        
        # Verificare cores ÎNAINTE
        total_cores = cpu_count()
        cpu_cores_to_use = max(1, total_cores - 3)
        
        print(f"💻 Total CPU cores: {total_cores}")
        print(f"🚀 GPU Thread: 1 RNG (xorshift_simple - kernel CORECT)")
        print(f"💻 CPU Thread: 20 RNG-uri ({cpu_cores_to_use} cores, lasă 3 pentru GPU)")
        print(f"🎯 Reverse Engineering: 6 LCG (INSTANT)")
        print(f"⚡ GPU + CPU pornesc SIMULTAN (threading paralel)!\n")
        
        # Load
        if last_n:
            print(f"📊 Ultimele {last_n} extrageri...")
            data = self.load_data(last_n=last_n)
        else:
            print(f"📊 {start_year}-{end_year}...")
            data = self.load_data(start_year=start_year, end_year=end_year)
        
        print(f"✅ {len(data)} extrageri\n")
        
        for i, e in enumerate(data, 1):
            print(f"  {i}. {e['data']:15s} → {e['numere']}")
        print()
        
        # RNG Testing - GPU Thread + CPU Multiprocessing PARALEL
        print(f"{'='*70}")
        print(f"  RNG TESTING - GPU Thread || CPU Multiprocessing")
        print(f"{'='*70}\n")
        
        results_queue = Queue()
        
        # GPU Thread
        gpu_thread = threading.Thread(
            target=gpu_thread_worker,
            args=(data, self.config, seed_range, results_queue)
        )
        
        # CPU Thread
        cpu_thread = threading.Thread(
            target=cpu_multiprocessing_worker,
            args=(data, self.config, seed_range, search_size, min_success_rate, results_queue)
        )
        
        # Start PARALEL!
        print("🚀 PORNIRE SIMULTANĂ GPU + CPU...\n")
        gpu_thread.start()
        cpu_thread.start()
        
        # Wait
        gpu_thread.join()
        cpu_thread.join()
        
        print("\n✅ AMBELE COMPLETE!\n")
        
        # Colectează
        rng_results = {}
        while not results_queue.empty():
            source, results = results_queue.get()
            print(f"📊 {source.upper()}: {len(results)} RNG-uri găsite")
            rng_results.update(results)
        
        if not rng_results:
            print("❌ Niciun RNG nu a trecut!")
            return
        
        # Pattern Analysis
        print(f"\n{'='*70}")
        print(f"  PATTERN ANALYSIS")
        print(f"{'='*70}\n")
        
        predictions = []
        
        for rng_name, result in sorted(rng_results.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            print(f"\n{rng_name.upper()} ({result['success_rate']:.1%}):")
            
            pattern = analyze_patterns_parallel_gpu_cpu(result['seeds'])
            
            # Afișare pattern-uri
            if pattern.get('top_patterns') and len(pattern['top_patterns']) > 1:
                # MULTIPLE PATTERN-URI CU 100%!
                print(f"  🔥 MULTIPLE PATTERN-URI PERFECTE (100% confidence):")
                for i, p in enumerate(pattern['top_patterns'], 1):
                    print(f"    {i}. {p['name'].upper()}")
                    print(f"       Formula: {p['formula']}")
                    print(f"       Seed prezis: {p['pred']:,}")
                    print(f"       Error: {p['error']}")
                print()
            elif pattern.get('top_patterns'):
                # UN SINGUR PATTERN (cel mai bun)
                p = pattern['top_patterns'][0]
                print(f"  🏆 BEST PATTERN: {p['name'].upper()}")
                print(f"  📐 Formula: {p['formula']}")
                print(f"  🎯 Confidence: {p['confidence']:.2f}%")
                print(f"  ❌ Error: {p['error']}")
            
            print(f"\n  📊 Toate patterns ({len(pattern.get('all_patterns', {}))}):")
            for pn, pd in pattern.get('all_patterns', {}).items():
                err_str = f"{pd.get('error', '?'):.2f}" if pd.get('error') != float('inf') else "∞"
                print(f"    {pn:20s}: error={err_str}")
            print()
            
            if pattern['predicted_seed']:
                try:
                    rng = create_rng(rng_name, pattern['predicted_seed'])
                    nums = generate_numbers(rng, self.config.numbers_to_draw, self.config.min_number, self.config.max_number)
                    print(f"  🎯 Seed: {pattern['predicted_seed']:,}")
                    print(f"  🎯 NUMERE: {sorted(nums)}")
                    
                    predictions.append({
                        'rng': rng_name,
                        'success_rate': result['success_rate'],
                        'pattern': pattern['pattern_type'],
                        'formula': pattern['formula'],
                        'confidence': pattern['confidence'],
                        'seed': pattern['predicted_seed'],
                        'numbers': sorted(nums)
                    })
                except Exception as e:
                    print(f"  ❌ Eroare predicție: {e}")
        
        # Save
        if predictions:
            output = f"gpu_safe_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output, 'w') as f:
                json.dump({'lottery': self.lottery_type, 'predictions': predictions}, f, indent=2)
            print(f"\n💾 Salvat: {output}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU-Safe Predictor')
    parser.add_argument('--lottery', default='5-40', choices=['5-40', '6-49', 'joker'])
    parser.add_argument('--last-n', type=int)
    parser.add_argument('--start-year', type=int)
    parser.add_argument('--end-year', type=int)
    parser.add_argument('--seed-range', type=int, nargs=2, default=[0, 100000000])
    parser.add_argument('--search-size', type=int, default=10000000)
    parser.add_argument('--min-success-rate', type=float, default=0.66)
    
    args = parser.parse_args()
    
    if not args.last_n and not (args.start_year and args.end_year):
        print("❌ Specifică --last-n SAU (--start-year și --end-year)!")
        sys.exit(1)
    
    predictor = GPUSafePredictor(args.lottery)
    predictor.run_prediction(
        last_n=args.last_n,
        start_year=args.start_year,
        end_year=args.end_year,
        seed_range=tuple(args.seed_range),
        search_size=args.search_size,
        min_success_rate=args.min_success_rate
    )
