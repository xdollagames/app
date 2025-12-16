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
    """GPU Thread - testează RNG-uri SECVENȚIAL dar cu TOT GPU-ul"""
    try:
        import cupy as cp
        print("🚀 [GPU Thread] CuPy importat cu succes!")
        
        # RNG-uri pentru GPU
        gpu_rngs_to_test = ['xorshift_simple']  # Adaugă altele când ai kernels
        
        print(f"🚀 [GPU] Va testa {len(gpu_rngs_to_test)} RNG-uri SECVENȚIAL (folosește TOT GPU-ul)\n")
        
        # CUDA Kernel simplu pentru xorshift
        test_kernel = cp.RawKernel(r'''
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
                
                # GPU batch testing
                batch_size = 2000000
                max_batches = 50
                
                for batch_num in range(max_batches):
                    seeds_batch = cp.random.randint(0, seed_range[1], size=batch_size, dtype=cp.uint32)
                    target_gpu = cp.array(target_sorted, dtype=cp.int32)
                    results_gpu = cp.zeros(batch_size, dtype=cp.int32)
                    
                    threads = 256
                    blocks = (batch_size + threads - 1) // threads
                    
                    test_kernel((blocks,), (threads,), 
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
            
            if success_rate >= 0.66:  # 66% threshold
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
    """CPU Thread - testează RNG-uri SECVENȚIAL cu TOATE cores-urile (multiprocessing spawn)"""
    num_cores = cpu_count()
    
    print(f"💻 [CPU Thread] Folosește TOATE cele {num_cores} cores")
    
    # RNG-uri CPU (exclude cele testate pe GPU)
    cpu_rngs = [r for r in RNG_TYPES.keys() if r != 'xorshift_simple']
    
    print(f"💻 [CPU] Va testa {len(cpu_rngs)} RNG-uri SECVENȚIAL (fiecare cu {num_cores} cores)\n")
    
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


def analyze_patterns_with_gpu(seeds):
    """Pattern analysis - folosește CuPy dacă e disponibil"""
    if len(seeds) < 3:
        return {'pattern_type': 'insufficient', 'predicted_seed': None, 'confidence': 0}
    
    x = np.arange(len(seeds))
    y = np.array(seeds)
    patterns = {}
    
    # Încearcă cu GPU
    try:
        import cupy as cp
        x_gpu = cp.asarray(x, dtype=cp.float64)
        y_gpu = cp.asarray(y, dtype=cp.float64)
        
        for deg in [1, 2, 3, 4]:
            if len(seeds) >= deg + 1:
                try:
                    coeffs = cp.polyfit(x_gpu, y_gpu, deg)
                    pred = float(cp.asnumpy(cp.poly1d(coeffs)(len(seeds))))
                    error = float(cp.asnumpy(cp.mean(cp.abs(y_gpu - cp.poly1d(coeffs)(x_gpu)))))
                    name = 'linear' if deg == 1 else f'poly_{deg}'
                    patterns[name] = {'pred': pred, 'error': error}
                except:
                    pass
    except:
        # Fallback CPU
        try:
            coeffs = np.polyfit(x, y, 1)
            pred = np.poly1d(coeffs)(len(seeds))
            patterns['linear'] = {'pred': pred, 'error': np.mean(np.abs(y - np.poly1d(coeffs)(x)))}
        except:
            pass
    
    # LCG Chain (IMPORTANT!)
    if len(seeds) >= 2:
        try:
            m = 2147483648
            X = np.array([[seeds[i-1], 1] for i in range(1, len(seeds))])
            Y = np.array([seeds[i] for i in range(1, len(seeds))])
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            a, c = coeffs
            lcg_pred = (a * seeds[-1] + c) % m
            errors = [abs((a * seeds[i-1] + c) % m - seeds[i]) for i in range(1, len(seeds))]
            patterns['lcg_chain'] = {'pred': lcg_pred, 'error': np.mean(errors), 'formula': f'S(n+1)=({a:.4f}*S(n)+{c:.2f})mod{m}'}
        except:
            pass
    
    # Modular
    if len(seeds) >= 2:
        try:
            diffs = np.diff(seeds)
            avg_diff = np.mean(diffs)
            m = 2147483648
            mod_pred = (seeds[-1] + avg_diff) % m
            errors = [abs((seeds[i-1] + avg_diff) % m - seeds[i]) for i in range(1, len(seeds))]
            patterns['modular'] = {'pred': mod_pred, 'error': np.mean(errors), 'formula': f'S(n+1)=(S(n)+{avg_diff:.2f})mod{m}'}
        except:
            pass
    
    # Best
    valid = {k: v for k, v in patterns.items() if v.get('pred') and v.get('error') != float('inf')}
    if not valid:
        return {'pattern_type': 'none', 'predicted_seed': None, 'confidence': 0}
    
    best_name = min(valid, key=lambda k: valid[k]['error'])
    best = valid[best_name]
    confidence = max(0, min(100, 100 * (1 - best['error'] / np.mean(y)))) if np.mean(y) > 0 else 0
    
    return {
        'pattern_type': best_name,
        'predicted_seed': int(round(best['pred'])),
        'confidence': round(confidence, 2),
        'formula': best.get('formula', best_name),
        'error': round(best['error'], 2),
        'all_patterns': patterns
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
        
        num_cores = cpu_count()
        print(f"🚀 GPU Thread: Va testa 1 RNG (xorshift_simple) cu TOT GPU-ul")
        print(f"💻 CPU Thread: Va testa 20 RNG-uri SECVENȚIAL, fiecare cu TOATE cele {num_cores} cores")
        print(f"🎯 Reverse Engineering: 6 LCG variants (INSTANT)")
        print(f"⚡ GPU + CPU rulează în PARALEL (threading)!\n")
        
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
            
            pattern = analyze_patterns_with_gpu(result['seeds'])
            
            print(f"  Pattern: {pattern['pattern_type']}")
            print(f"  Formula: {pattern.get('formula', 'N/A')}")
            print(f"  Confidence: {pattern['confidence']:.2f}%")
            
            if 'all_patterns' in pattern:
                print(f"  Toate patterns:")
                for pn, pd in pattern['all_patterns'].items():
                    print(f"    {pn}: error={pd.get('error', '?')}")
            
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
