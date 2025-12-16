#!/usr/bin/env python3
"""
GPU-ACCELERATED ULTIMATE PREDICTOR
Folosește GPU pentru testarea rapidă a seed-urilor
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

# Check GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU detectat! Se va folosi accelerare CUDA")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  CuPy nu e instalat. Se va folosi CPU multicore")
    import numpy as cp

from lottery_config import get_lottery_config
from advanced_rng_library import RNG_TYPES, create_rng, generate_numbers


# GPU Kernels pentru RNG-uri simple
if GPU_AVAILABLE:
    xorshift_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void xorshift_simple_batch(
        const unsigned int* seeds,
        const int num_seeds,
        const int numbers_to_draw,
        const int min_number,
        const int max_number,
        unsigned int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        unsigned int state = seeds[idx];
        
        for (int i = 0; i < numbers_to_draw; i++) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            
            int range_size = max_number - min_number + 1;
            int num = min_number + (state % range_size);
            results[idx * numbers_to_draw + i] = num;
        }
    }
    ''', 'xorshift_simple_batch')
    
    lcg_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void lcg_glibc_batch(
        const unsigned int* seeds,
        const int num_seeds,
        const int numbers_to_draw,
        const int min_number,
        const int max_number,
        unsigned int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        unsigned long long state = seeds[idx] % 2147483648ULL;
        
        for (int i = 0; i < numbers_to_draw; i++) {
            state = (1103515245ULL * state + 12345ULL) % 2147483648ULL;
            
            int range_size = max_number - min_number + 1;
            int num = min_number + (state % range_size);
            results[idx * numbers_to_draw + i] = num;
        }
    }
    ''', 'lcg_glibc_batch')
    
    java_random_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void java_random_batch(
        const unsigned long long* seeds,
        const int num_seeds,
        const int numbers_to_draw,
        const int min_number,
        const int max_number,
        unsigned int* results
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_seeds) return;
        
        unsigned long long state = (seeds[idx] ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1);
        
        for (int i = 0; i < numbers_to_draw; i++) {
            state = (state * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
            
            int range_size = max_number - min_number + 1;
            int num = min_number + ((state >> 16) % range_size);
            results[idx * numbers_to_draw + i] = num;
        }
    }
    ''', 'java_random_batch')


def gpu_test_seeds_batch(rng_type: str, seeds: np.ndarray, target: List[int], 
                         numbers_to_draw: int, min_number: int, max_number: int) -> Optional[int]:
    """Testează un batch de seeds pe GPU"""
    if not GPU_AVAILABLE:
        return None
    
    num_seeds = len(seeds)
    target_sorted = sorted(target)
    
    # Transferă pe GPU
    seeds_gpu = cp.array(seeds, dtype=cp.uint32)
    results_gpu = cp.zeros((num_seeds, numbers_to_draw), dtype=cp.uint32)
    
    threads_per_block = 256
    blocks = (num_seeds + threads_per_block - 1) // threads_per_block
    
    try:
        if rng_type == 'xorshift_simple':
            xorshift_kernel(
                (blocks,), (threads_per_block,),
                (seeds_gpu, num_seeds, numbers_to_draw, min_number, max_number, results_gpu)
            )
        elif rng_type == 'lcg_glibc':
            lcg_kernel(
                (blocks,), (threads_per_block,),
                (seeds_gpu, num_seeds, numbers_to_draw, min_number, max_number, results_gpu)
            )
        elif rng_type == 'java_random':
            seeds_gpu_64 = cp.array(seeds, dtype=cp.uint64)
            java_random_kernel(
                (blocks,), (threads_per_block,),
                (seeds_gpu_64, num_seeds, numbers_to_draw, min_number, max_number, results_gpu)
            )
        else:
            return None  # RNG nu e suportat pe GPU
        
        # Transferă rezultatele înapoi
        results_cpu = cp.asnumpy(results_gpu)
        
        # Caută match
        for i, result in enumerate(results_cpu):
            if sorted(result) == target_sorted:
                return int(seeds[i])
        
        return None
    except:
        return None


def cpu_find_seed_worker(args):
    """Worker CPU pentru RNG-uri care nu au suport GPU"""
    draw_idx, numbers, rng_type, lottery_config, seed_range, search_size = args
    target_sorted = sorted(numbers)
    
    # Timeout pentru Mersenne
    import time
    start_time = time.time()
    timeout_seconds = 30 if rng_type == 'mersenne' else 999999
    
    if rng_type == 'mersenne':
        search_size = min(search_size, 50000)
    
    test_seeds = random.sample(range(seed_range[0], seed_range[1]), 
                              min(search_size, seed_range[1] - seed_range[0]))
    
    for seed in test_seeds:
        if rng_type == 'mersenne' and (time.time() - start_time) > timeout_seconds:
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
        except:
            continue
    
    return (draw_idx, None)


def analyze_seed_pattern(seeds: List[int]) -> Dict:
    """Analizează pattern-ul matematic - versiune optimizată"""
    if len(seeds) < 3:
        return {
            'pattern_type': 'insufficient_data',
            'predicted_seed': None,
            'confidence': 0,
            'formula': 'N/A'
        }
    
    x = np.arange(len(seeds))
    y = np.array(seeds)
    all_patterns = {}
    
    # 1. LCG CHAIN (cel mai probabil pentru loterie)
    if len(seeds) >= 2:
        try:
            m_estimate = max(seeds) * 2 if max(seeds) > 0 else 2147483648
            X = np.array([[seeds[i-1], 1] for i in range(1, len(seeds))])
            Y = np.array([seeds[i] for i in range(1, len(seeds))])
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            a, c = coeffs
            
            lcg_pred = (a * seeds[-1] + c) % m_estimate
            errors = [abs((a * seeds[i-1] + c) % m_estimate - seeds[i]) for i in range(1, len(seeds))]
            lcg_error = np.mean(errors) if errors else float('inf')
            
            all_patterns['lcg_chain'] = {
                'pred': lcg_pred,
                'error': lcg_error,
                'formula': f"S(n+1) = ({a:.4f}*S(n) + {c:.2f}) mod {int(m_estimate)}"
            }
        except:
            all_patterns['lcg_chain'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # 2. MODULAR
    if len(seeds) >= 2:
        try:
            diffs = np.diff(seeds)
            avg_diff = np.mean(diffs)
            m_estimate = max(seeds) * 2 if max(seeds) > 0 else 2147483648
            
            modular_pred = (seeds[-1] + avg_diff) % m_estimate
            errors = [abs((seeds[i-1] + avg_diff) % m_estimate - seeds[i]) for i in range(1, len(seeds))]
            modular_error = np.mean(errors)
            
            all_patterns['modular'] = {
                'pred': modular_pred,
                'error': modular_error,
                'formula': f"S(n+1) = (S(n) + {avg_diff:.2f}) mod {int(m_estimate)}"
            }
        except:
            all_patterns['modular'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # 3. LINEAR
    try:
        linear_coeffs = np.polyfit(x, y, 1)
        linear_pred = np.poly1d(linear_coeffs)(len(seeds))
        linear_error = np.mean(np.abs(y - np.poly1d(linear_coeffs)(x)))
        all_patterns['linear'] = {
            'pred': linear_pred,
            'error': linear_error,
            'formula': f"y = {linear_coeffs[0]:.2f}*x + {linear_coeffs[1]:.2f}"
        }
    except:
        all_patterns['linear'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # 4. CONSTANT DIFFERENCE
    if len(seeds) >= 2:
        try:
            diffs = np.diff(seeds)
            avg_diff = np.mean(diffs)
            const_diff_pred = seeds[-1] + avg_diff
            const_diff_error = np.std(diffs)
            
            all_patterns['const_diff'] = {
                'pred': const_diff_pred,
                'error': const_diff_error,
                'formula': f"S(n+1) = S(n) + {avg_diff:.2f}"
            }
        except:
            all_patterns['const_diff'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # Selectare best pattern
    valid_patterns = {k: v for k, v in all_patterns.items() 
                     if v['pred'] is not None and not np.isnan(v['error']) and v['error'] != float('inf')}
    
    if not valid_patterns:
        return {
            'pattern_type': 'no_valid_pattern',
            'predicted_seed': None,
            'confidence': 0,
            'formula': 'N/A'
        }
    
    best_pattern_name = min(valid_patterns, key=lambda k: valid_patterns[k]['error'])
    best_pattern = valid_patterns[best_pattern_name]
    
    predicted_seed = int(round(best_pattern['pred']))
    mean_seed = np.mean(y)
    confidence = max(0, min(100, 100 * (1 - best_pattern['error'] / mean_seed))) if mean_seed > 0 else 0
    
    return {
        'pattern_type': best_pattern_name,
        'predicted_seed': predicted_seed,
        'confidence': round(confidence, 2),
        'formula': best_pattern['formula'],
        'error': round(best_pattern['error'], 2)
    }


class GPUPredictor:
    # RNG-uri cu suport GPU
    GPU_SUPPORTED_RNGS = ['xorshift_simple', 'lcg_glibc', 'java_random']
    
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
    
    def test_rng_gpu(self, rng_name: str, data: List[Dict], 
                    seed_range: tuple = (0, 10000000), 
                    batch_size: int = 1000000):
        """Testează un RNG pe GPU cu batch processing"""
        print(f"🚀 GPU Mode: Testare {rng_name} cu batch size {batch_size:,}")
        
        seeds_found = []
        draws_with_seeds = []
        
        for idx, entry in enumerate(data):
            numbers = entry.get('numere', [])
            if len(numbers) != self.config.numbers_to_draw:
                continue
            
            # Generează batch de seeds
            seeds_to_test = np.random.randint(seed_range[0], seed_range[1], size=batch_size, dtype=np.uint32)
            
            # Testează pe GPU
            found_seed = gpu_test_seeds_batch(
                rng_name, seeds_to_test, numbers,
                self.config.numbers_to_draw,
                self.config.min_number,
                self.config.max_number
            )
            
            if found_seed is not None:
                seeds_found.append(found_seed)
                draws_with_seeds.append({
                    'idx': idx,
                    'date': entry['data'],
                    'numbers': numbers,
                    'seed': found_seed
                })
            
            print(f"  [{idx+1}/{len(data)}] Seeds găsite: {len(seeds_found)}", end='\r')
        
        print(f"\n✅ Total seeds: {len(seeds_found)}/{len(data)} ({len(seeds_found)/len(data):.1%})")
        
        return seeds_found, draws_with_seeds
    
    def test_rng_cpu(self, rng_name: str, data: List[Dict], 
                    seed_range: tuple, search_size: int):
        """Testează un RNG pe CPU multicore"""
        print(f"💻 CPU Mode: Testare {rng_name}")
        
        tasks = []
        for i, entry in enumerate(data):
            numbers = entry.get('numere', [])
            if len(numbers) == self.config.numbers_to_draw:
                tasks.append((i, numbers, rng_name, self.config, seed_range, search_size))
        
        seeds_found = []
        draws_with_seeds = []
        num_cores = cpu_count()
        
        with Pool(processes=num_cores) as pool:
            optimal_chunksize = max(1, len(tasks) // (num_cores * 4))
            for i, result in enumerate(pool.imap_unordered(cpu_find_seed_worker, tasks, chunksize=optimal_chunksize)):
                idx, seed = result
                
                if seed is not None:
                    seeds_found.append(seed)
                    draws_with_seeds.append({
                        'idx': idx,
                        'date': data[idx]['data'],
                        'numbers': data[idx]['numere'],
                        'seed': seed
                    })
                
                if (i + 1) % 5 == 0 or (i + 1) == len(tasks):
                    print(f"  [{i + 1}/{len(tasks)}] Seeds găsite: {len(seeds_found)}", end='\r')
        
        print(f"\n✅ Total seeds: {len(seeds_found)}/{len(data)} ({len(seeds_found)/len(data):.1%})")
        
        return seeds_found, draws_with_seeds
    
    def run_prediction(self, last_n: Optional[int] = None,
                      start_year: Optional[int] = None,
                      end_year: Optional[int] = None,
                      seed_range: tuple = (0, 10000000),
                      search_size: int = 2000000,
                      min_success_rate: float = 0.5):
        """Rulează predicția cu GPU/CPU"""
        
        print(f"\n{'='*70}")
        print(f"  🚀 GPU-ACCELERATED PREDICTOR - {self.lottery_type.upper()}")
        print(f"{'='*70}\n")
        
        if GPU_AVAILABLE:
            print("✅ GPU Mode: ACTIVAT")
        else:
            print("⚠️  CPU Mode: GPU nu e disponibil")
        
        print()
        
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
        
        # Test RNG-uri
        print(f"{'='*70}")
        print(f"  TESTARE RNG-URI")
        print(f"{'='*70}\n")
        
        rng_results = {}
        
        for rng_name in RNG_TYPES.keys():
            print(f"\n{'='*70}")
            print(f"RNG: {rng_name.upper()}")
            print(f"{'='*70}")
            
            # Folosește GPU pentru RNG-uri suportate
            if GPU_AVAILABLE and rng_name in self.GPU_SUPPORTED_RNGS:
                seeds, draws = self.test_rng_gpu(rng_name, data, seed_range, batch_size=1000000)
            else:
                seeds, draws = self.test_rng_cpu(rng_name, data, seed_range, search_size)
            
            success_rate = len(seeds) / len(data) if len(data) > 0 else 0
            
            if success_rate >= min_success_rate:
                print(f"✅ SUCCESS ({success_rate:.1%} >= {min_success_rate:.1%})")
                rng_results[rng_name] = {
                    'seeds': seeds,
                    'draws': draws,
                    'success_rate': success_rate
                }
            else:
                print(f"❌ SKIP ({success_rate:.1%} < {min_success_rate:.1%})")
        
        if not rng_results:
            print(f"\n❌ Niciun RNG nu a trecut!")
            return
        
        # Analiză și predicții
        print(f"\n{'='*70}")
        print(f"  PREDICȚII")
        print(f"{'='*70}\n")
        
        predictions = []
        
        for rng_name, result in sorted(rng_results.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            print(f"\n{'='*70}")
            print(f"RNG: {rng_name.upper()} ({result['success_rate']:.1%})")
            print(f"{'='*70}")
            
            pattern = analyze_seed_pattern(result['seeds'])
            
            print(f"📊 Pattern: {pattern['pattern_type'].upper()}")
            print(f"📐 Formula: {pattern['formula']}")
            print(f"🎯 Confidence: {pattern['confidence']:.2f}%\n")
            
            if pattern['predicted_seed'] is not None:
                try:
                    rng = create_rng(rng_name, pattern['predicted_seed'])
                    predicted_numbers = generate_numbers(
                        rng, self.config.numbers_to_draw,
                        self.config.min_number, self.config.max_number
                    )
                    
                    print(f"🎯 Seed prezis: {pattern['predicted_seed']:,}")
                    print(f"🎯 NUMERE PREZISE: {sorted(predicted_numbers)}\n")
                    
                    predictions.append({
                        'rng': rng_name,
                        'success_rate': result['success_rate'],
                        'pattern': pattern['pattern_type'],
                        'formula': pattern['formula'],
                        'confidence': pattern['confidence'],
                        'seed': pattern['predicted_seed'],
                        'numbers': sorted(predicted_numbers)
                    })
                except Exception as e:
                    print(f"❌ Eroare: {e}\n")
        
        # Sumar
        if predictions:
            print(f"\n{'='*70}")
            print(f"  SUMAR FINAL")
            print(f"{'='*70}\n")
            
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. {pred['rng'].upper()}")
                print(f"   Success: {pred['success_rate']:.1%} | Confidence: {pred['confidence']:.1f}%")
                print(f"   Pattern: {pred['pattern']}")
                print(f"   Numere: {pred['numbers']}\n")
            
            output_file = f"gpu_prediction_{self.lottery_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'lottery': self.lottery_type,
                    'timestamp': datetime.now().isoformat(),
                    'gpu_used': GPU_AVAILABLE,
                    'data_size': len(data),
                    'predictions': predictions
                }, f, indent=2)
            
            print(f"💾 Salvat: {output_file}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU-Accelerated Predictor')
    parser.add_argument('--lottery', default='5-40', choices=['5-40', '6-49', 'joker'])
    parser.add_argument('--last-n', type=int, help='Ultimele N extrageri')
    parser.add_argument('--start-year', type=int)
    parser.add_argument('--end-year', type=int)
    parser.add_argument('--seed-range', type=int, nargs=2, default=[0, 10000000])
    parser.add_argument('--search-size', type=int, default=2000000)
    parser.add_argument('--min-success-rate', type=float, default=0.5)
    
    args = parser.parse_args()
    
    if not args.last_n and not (args.start_year and args.end_year):
        print("❌ Specifică --last-n SAU (--start-year și --end-year)!")
        sys.exit(1)
    
    predictor = GPUPredictor(args.lottery)
    predictor.run_prediction(
        last_n=args.last_n,
        start_year=args.start_year,
        end_year=args.end_year,
        seed_range=tuple(args.seed_range),
        search_size=args.search_size,
        min_success_rate=args.min_success_rate
    )
