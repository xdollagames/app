#!/usr/bin/env python3
"""
SIMPLIFIED MAX PREDICTOR - CPU FOCUS cu Pattern Analysis
Fără GPU kernels complexi - doar reverse engineering și CPU multicore
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

from lottery_config import get_lottery_config
from advanced_rng_library import RNG_TYPES, create_rng, generate_numbers


# Import condiționat CuPy DOAR pentru pattern analysis
try:
    import cupy as cp
    GPU_FOR_PATTERNS = True
    print("✅ GPU (CuPy) disponibil pentru Pattern Analysis")
except:
    import numpy as cp
    GPU_FOR_PATTERNS = False
    print("⚠️  CuPy nu e disponibil - CPU mode")


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
        return None
    return (x % m + m) % m


def reverse_lcg_generic(output_number: int, min_number: int, max_number: int, a: int, c: int, m: int) -> Optional[int]:
    """INVERSĂ GENERICĂ pentru orice LCG"""
    range_size = max_number - min_number + 1
    target_mod = output_number - min_number
    
    a_inv = compute_modular_inverse(a, m)
    if a_inv is None:
        return None
    
    for k in range(0, min(1000, m // range_size + 1)):
        state = target_mod + k * range_size
        if state >= m:
            break
        prev_state = (a_inv * (state - c)) % m
        if 0 <= prev_state < m:
            return prev_state
    return None


def try_reverse_engineering(rng_name: str, numbers: List[int], lottery_config) -> Optional[int]:
    """Reverse engineering pentru 16 RNG-uri"""
    if not numbers:
        return None
    
    # LCG variants
    if rng_name == 'lcg_glibc':
        return reverse_lcg_generic(numbers[0], lottery_config.min_number, lottery_config.max_number, 1103515245, 12345, 2**31)
    elif rng_name == 'lcg_minstd':
        return reverse_lcg_generic(numbers[0], lottery_config.min_number, lottery_config.max_number, 48271, 0, 2**31 - 1)
    elif rng_name == 'lcg_randu':
        return reverse_lcg_generic(numbers[0], lottery_config.min_number, lottery_config.max_number, 65539, 0, 2**31)
    elif rng_name == 'lcg_borland':
        return reverse_lcg_generic(numbers[0], lottery_config.min_number, lottery_config.max_number, 22695477, 1, 2**32)
    elif rng_name == 'lcg_weak':
        return reverse_lcg_generic(numbers[0], lottery_config.min_number, lottery_config.max_number, 9301, 49297, 233280)
    elif rng_name == 'php_rand':
        return reverse_lcg_generic(numbers[0], lottery_config.min_number, lottery_config.max_number, 1103515245, 12345, 0x7FFFFFFF)
    
    # Altele - skip pentru simplificare (vor folosi brute force)
    return None


def find_seed_worker(args):
    """Worker CPU - cu reverse engineering"""
    draw_idx, numbers, rng_name, lottery_config, seed_range, search_size = args
    target_sorted = sorted(numbers)
    
    # Încercăm REVERSE mai întâi
    reversed_seed = try_reverse_engineering(rng_name, numbers, lottery_config)
    if reversed_seed is not None:
        # Verificăm
        try:
            rng = create_rng(rng_name, reversed_seed)
            generated = generate_numbers(rng, lottery_config.numbers_to_draw, lottery_config.min_number, lottery_config.max_number)
            if sorted(generated) == target_sorted:
                return (draw_idx, reversed_seed)
        except:
            pass
    
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


def analyze_patterns_gpu_accelerated(seeds: List[int]) -> Dict:
    """Pattern analysis cu GPU (CuPy) pentru polyfit"""
    if len(seeds) < 3:
        return {'pattern_type': 'insufficient_data', 'predicted_seed': None, 'confidence': 0}
    
    x = np.arange(len(seeds))
    y = np.array(seeds)
    
    all_patterns = {}
    
    # Polynomial fits cu GPU acceleration
    if GPU_FOR_PATTERNS:
        try:
            x_gpu = cp.asarray(x, dtype=cp.float64)
            y_gpu = cp.asarray(y, dtype=cp.float64)
            
            for degree in [1, 2, 3, 4]:
                if len(seeds) >= degree + 1:
                    try:
                        coeffs = cp.polyfit(x_gpu, y_gpu, degree)
                        pred = float(cp.asnumpy(cp.poly1d(coeffs)(len(seeds))))
                        error = float(cp.asnumpy(cp.mean(cp.abs(y_gpu - cp.poly1d(coeffs)(x_gpu)))))
                        name = 'linear' if degree == 1 else f'poly_{degree}'
                        all_patterns[name] = {'pred': pred, 'error': error, 'formula': f'poly(deg={degree})'}
                    except:
                        pass
        except:
            pass
    
    # LCG Chain (cel mai important!)
    if len(seeds) >= 2:
        try:
            m = 2147483648
            X = np.array([[seeds[i-1], 1] for i in range(1, len(seeds))])
            Y = np.array([seeds[i] for i in range(1, len(seeds))])
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            a, c = coeffs
            lcg_pred = (a * seeds[-1] + c) % m
            errors = [abs((a * seeds[i-1] + c) % m - seeds[i]) for i in range(1, len(seeds))]
            all_patterns['lcg_chain'] = {'pred': lcg_pred, 'error': np.mean(errors), 'formula': f'S(n+1)=({a:.4f}*S(n)+{c:.2f})mod{m}'}
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
            all_patterns['modular'] = {'pred': mod_pred, 'error': np.mean(errors), 'formula': f'S(n+1)=(S(n)+{avg_diff:.2f})mod{m}'}
        except:
            pass
    
    # Best pattern
    valid = {k: v for k, v in all_patterns.items() if v['pred'] is not None and v['error'] != float('inf')}
    
    if not valid:
        return {'pattern_type': 'none', 'predicted_seed': None, 'confidence': 0, 'all_patterns': all_patterns}
    
    best_name = min(valid, key=lambda k: valid[k]['error'])
    best = valid[best_name]
    
    confidence = max(0, min(100, 100 * (1 - best['error'] / np.mean(y)))) if np.mean(y) > 0 else 0
    
    return {
        'pattern_type': best_name,
        'predicted_seed': int(round(best['pred'])),
        'confidence': round(confidence, 2),
        'formula': best['formula'],
        'error': round(best['error'], 2),
        'all_patterns': all_patterns
    }


class SimplifiedPredictor:
    def __init__(self, lottery_type: str = "5-40"):
        self.lottery_type = lottery_type
        self.config = get_lottery_config(lottery_type)
        self.data_file = f"{lottery_type}_data.json"
    
    def load_data(self, last_n: Optional[int] = None, start_year: Optional[int] = None, end_year: Optional[int] = None):
        """Încarcă date"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"❌ {self.data_file} nu există!")
            sys.exit(1)
        
        all_data = data['draws'] if 'draws' in data else data
        
        if last_n:
            filtered = all_data[-last_n:]
        elif start_year and end_year:
            filtered = []
            for entry in all_data:
                try:
                    date_str = entry.get('data', entry.get('date', ''))
                    year = int(date_str.split('.')[-1]) if '.' in date_str else int(date_str.split('-')[0])
                    if start_year <= year <= end_year:
                        filtered.append(entry)
                except:
                    continue
        else:
            filtered = all_data
        
        return [{'data': e.get('data', e.get('date')), 'numere': e.get('numere', e.get('numbers', e.get('numbers_sorted')))} for e in filtered]
    
    def run_prediction(self, last_n=None, start_year=None, end_year=None, 
                      seed_range=(0, 100000000), search_size=10000000, min_success_rate=0.66):
        """Rulează predicția"""
        
        print(f"\n{'='*70}")
        print(f"  SIMPLIFIED MAX PREDICTOR - {self.lottery_type.upper()}")
        print(f"{'='*70}\n")
        
        num_cores = cpu_count()
        print(f"💻 CPU Cores: {num_cores}")
        print(f"🎯 Reverse Engineering: 6 LCG variants")
        print(f"🔍 Seed Range: {seed_range[0]:,} - {seed_range[1]:,}")
        print(f"📊 Search Size: {search_size:,} seeds\n")
        
        # Load data
        if last_n:
            print(f"📊 Încărcare ultimele {last_n} extrageri...")
            data = self.load_data(last_n=last_n)
        else:
            print(f"📊 Încărcare {start_year}-{end_year}...")
            data = self.load_data(start_year=start_year, end_year=end_year)
        
        print(f"✅ {len(data)} extrageri\n")
        
        # Afișează extrageri
        print("📋 Extrageri:")
        for i, e in enumerate(data, 1):
            print(f"  {i}. {e['data']:15s} → {e['numere']}")
        print()
        
        # Test RNG-uri
        print(f"{'='*70}")
        print(f"  RNG TESTING - CPU Multiprocessing ({num_cores} cores)")
        print(f"{'='*70}\n")
        
        rng_results = {}
        
        for idx, rng_name in enumerate(RNG_TYPES.keys(), 1):
            print(f"[{idx}/21] {rng_name.upper()}... ", end='', flush=True)
            
            tasks = [(i, e['numere'], rng_name, self.config, seed_range, search_size) 
                    for i, e in enumerate(data) if len(e['numere']) == self.config.numbers_to_draw]
            
            seeds_found = []
            draws_with_seeds = []
            
            with Pool(processes=num_cores) as pool:
                for i, result in enumerate(pool.imap_unordered(find_seed_worker, tasks)):
                    idx_task, seed = result
                    if seed is not None:
                        seeds_found.append(seed)
                        draws_with_seeds.append({'idx': idx_task, 'date': data[idx_task]['data'], 
                                                'numbers': data[idx_task]['numere'], 'seed': seed})
            
            success_rate = len(seeds_found) / len(data) if len(data) > 0 else 0
            print(f"{len(seeds_found)}/{len(data)} ({success_rate:.1%})")
            
            if success_rate >= min_success_rate:
                draws_with_seeds.sort(key=lambda x: x['idx'])
                seeds_found = [d['seed'] for d in draws_with_seeds]
                rng_results[rng_name] = {'seeds': seeds_found, 'draws': draws_with_seeds, 'success_rate': success_rate}
        
        if not rng_results:
            print(f"\n❌ Niciun RNG nu a trecut de {min_success_rate:.1%}!")
            return
        
        # Pattern analysis
        print(f"\n{'='*70}")
        print(f"  PATTERN ANALYSIS")
        print(f"{'='*70}\n")
        
        predictions = []
        
        for rng_name, result in sorted(rng_results.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            print(f"\n{rng_name.upper()} ({result['success_rate']:.1%}):")
            
            pattern = analyze_patterns_gpu_accelerated(result['seeds'])
            
            print(f"  Pattern: {pattern['pattern_type']}")
            print(f"  Formula: {pattern.get('formula', 'N/A')}")
            print(f"  Confidence: {pattern['confidence']:.2f}%")
            
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
                except:
                    pass
        
        # Salvare
        if predictions:
            output = f"prediction_{self.lottery_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output, 'w') as f:
                json.dump({'lottery': self.lottery_type, 'predictions': predictions}, f, indent=2)
            print(f"\n💾 Salvat: {output}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
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
    
    predictor = SimplifiedPredictor(args.lottery)
    predictor.run_prediction(
        last_n=args.last_n,
        start_year=args.start_year,
        end_year=args.end_year,
        seed_range=tuple(args.seed_range),
        search_size=args.search_size,
        min_success_rate=args.min_success_rate
    )
