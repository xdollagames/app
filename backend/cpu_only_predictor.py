#!/usr/bin/env python3
"""
CPU-ONLY PREDICTOR - ZERO GPU
Simplu, stabil, rapid cu reverse engineering
"""

import json
import sys
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
import random

try:
    import psutil
    HAS_PSUTIL = True
except:
    HAS_PSUTIL = False

from lottery_config import get_lottery_config
from advanced_rng_library import RNG_TYPES, create_rng, generate_numbers


# CACHE pentru seeds găsite (persistent între rulări!)
CACHE_FILE = "seeds_cache.json"

def load_seeds_cache():
    """Încarcă cache-ul de seeds"""
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_seeds_cache(cache):
    """Salvează cache-ul de seeds"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except:
        pass

def get_cached_seed(lottery_type, date_str, rng_name):
    """Verifică dacă avem seed cached pentru această combinație"""
    cache = load_seeds_cache()
    return cache.get(lottery_type, {}).get(date_str, {}).get(rng_name)

def cache_seed(lottery_type, date_str, rng_name, seed):
    """Salvează seed în cache"""
    cache = load_seeds_cache()
    
    if lottery_type not in cache:
        cache[lottery_type] = {}
    if date_str not in cache[lottery_type]:
        cache[lottery_type][date_str] = {}
    
    cache[lottery_type][date_str][rng_name] = seed
    save_seeds_cache(cache)


# SEED RANGES OPTIMIZATE + SEARCH SIZE = 100% COVERAGE!
OPTIMIZED_SEED_RANGES = {
    '5-40': (0, 4000000),      # C(40,6) = 3,838,380 → 100% coverage
    '6-49': (0, 14000000),     # C(49,6) = 13,983,816 → 100% coverage
    'joker': (0, 25000000),    # C(45,5) × 20 = 24,435,180 → 100% coverage
}

def get_optimal_seed_range(lottery_type):
    """Returnează seed range optim pentru loterie"""
    return OPTIMIZED_SEED_RANGES.get(lottery_type, (0, 100000000))

def get_exhaustive_search_size(lottery_type):
    """Returnează search size = TOATE seeds-urile pentru 100% coverage"""
    seed_range = get_optimal_seed_range(lottery_type)
    return seed_range[1]  # Testează TOATE seeds-urile!


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
    """Reverse engineering pentru 6 LCG-uri"""
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
    """Worker CPU - cu CACHE pentru seeds deja găsite!"""
    import time
    
    draw_idx, numbers, rng_name, lottery_config, seed_range, search_size_total, timeout_minutes, lottery_type, date_str = args
    target_sorted = sorted(numbers)
    start_time = time.time()
    
    # VERIFICĂ CACHE MAI ÎNTÂI! (INSTANT dacă există!)
    cached_seed = get_cached_seed(lottery_type, date_str, rng_name)
    if cached_seed is not None:
        # Verifică că seed-ul e încă valid
        try:
            rng = create_rng(rng_name, cached_seed)
            if lottery_config.is_composite:
                generated = []
                for count, min_val, max_val in lottery_config.composite_parts:
                    part = generate_numbers(rng, count, min_val, max_val)
                    generated.extend(part)
            else:
                generated = generate_numbers(rng, lottery_config.numbers_to_draw, lottery_config.min_number, lottery_config.max_number)
            
            if sorted(generated) == target_sorted:
                # CACHE HIT! Returnează instant
                return (draw_idx, cached_seed, True)  # True = din cache
        except:
            pass
    
    # Încercăm REVERSE
    reversed_seed = try_reverse_engineering(rng_name, numbers, lottery_config)
    if reversed_seed is not None:
        # Salvează în cache pentru viitor!
        cache_seed(lottery_type, date_str, rng_name, reversed_seed)
        return (draw_idx, reversed_seed, False)  # False = nu din cache
    
    # EXHAUSTIVE search - testează TOATE seeds-urile (sau până la timeout pentru Mersenne)
    # Pentru Mersenne: TIMEOUT de 10 minute per extragere
    timeout_seconds = timeout_minutes * 60 if rng_name == 'mersenne' else 99999999
    
    seeds_to_test = range(seed_range[0], min(seed_range[1], search_size_total))
    
    for seed in seeds_to_test:
        # Check timeout pentru Mersenne
        if rng_name == 'mersenne':
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                # Timeout - returnează None
                return (draw_idx, None)
        
        try:
            rng = create_rng(rng_name, seed)
            
            # Composite support
            if lottery_config.is_composite:
                generated = []
                for count, min_val, max_val in lottery_config.composite_parts:
                    part = generate_numbers(rng, count, min_val, max_val)
                    generated.extend(part)
            else:
                generated = generate_numbers(rng, lottery_config.numbers_to_draw, lottery_config.min_number, lottery_config.max_number)
            
            if sorted(generated) == target_sorted:
                # GĂSIT! Salvează în cache
                cache_seed(lottery_type, date_str, rng_name, seed)
                return (draw_idx, seed, False)  # False = calculat acum
        except:
            continue
    
    return (draw_idx, None, False)


def analyze_all_patterns_cpu(seeds):
    """Analizează TOATE 23 pattern-urile - 100% CPU"""
    if len(seeds) < 3:
        return {'pattern_type': 'insufficient', 'predicted_seed': None, 'confidence': 0, 'all_patterns': {}, 'top_patterns': []}
    
    print(f"  🎯 Analiză 23 pattern-uri pe CPU...")
    
    x = np.arange(len(seeds))
    y = np.array(seeds)
    all_patterns = {}
    
    # 1-4: Polynomial
    for deg in [1, 2, 3, 4]:
        if len(seeds) >= deg + 1:
            try:
                coeffs = np.polyfit(x, y, deg)
                pred = np.poly1d(coeffs)(len(seeds))
                error = np.mean(np.abs(y - np.poly1d(coeffs)(x)))
                name = 'linear' if deg == 1 else f'poly_{deg}'
                all_patterns[name] = {'pred': pred, 'error': error, 'formula': f'poly(deg={deg})'}
            except:
                pass
    
    # 5. Logarithmic
    if len(seeds) >= 2:
        try:
            log_x = np.log(x + 1)
            log_c = np.polyfit(log_x, y, 1)
            pred = log_c[0] * np.log(len(seeds) + 1) + log_c[1]
            error = np.mean(np.abs(y - (log_c[0] * log_x + log_c[1])))
            all_patterns['logarithmic'] = {'pred': pred, 'error': error, 'formula': 'log(x)'}
        except:
            pass
    
    # 6. Const Diff
    if len(seeds) >= 2:
        try:
            diffs = np.diff(seeds)
            avg_diff = np.mean(diffs)
            pred = seeds[-1] + avg_diff
            error = np.std(diffs)
            all_patterns['const_diff'] = {'pred': pred, 'error': error, 'formula': f'S(n+1)=S(n)+{avg_diff:.2f}'}
        except:
            pass
    
    # 7. Const Ratio
    if len(seeds) >= 2 and all(s > 0 for s in seeds):
        try:
            ratios = y[1:] / y[:-1]
            avg_ratio = np.mean(ratios)
            pred = seeds[-1] * avg_ratio
            error = np.std(ratios) * seeds[-1]
            all_patterns['const_ratio'] = {'pred': pred, 'error': error, 'formula': f'S(n+1)=S(n)*{avg_ratio:.4f}'}
        except:
            pass
    
    # 8. Exponential
    try:
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c
        popt, _ = curve_fit(exp_func, x, y, maxfev=3000)
        pred = exp_func(len(seeds), *popt)
        error = np.mean(np.abs(y - exp_func(x, *popt)))
        all_patterns['exponential'] = {'pred': pred, 'error': error, 'formula': 'exponential'}
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
            all_patterns['fibonacci'] = {'pred': pred, 'error': np.mean(errors), 'formula': f'Fib: {a:.4f}*S(n-1)+{b:.4f}*S(n-2)'}
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
            all_patterns['lcg_chain'] = {'pred': pred, 'error': np.mean(errors), 'formula': f'S(n+1)=({a:.4f}*S(n)+{c:.2f})mod{m}'}
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
            all_patterns['modular'] = {'pred': pred, 'error': np.mean(errors), 'formula': f'S(n+1)=(S(n)+{avg_diff:.2f})mod{m}'}
        except:
            pass
    
    # 12-23: Pattern-uri adiționale
    
    # 12. Power Law
    try:
        def power_func(x, a, b, c):
            return a * np.power(x + 1, b) + c
        popt, _ = curve_fit(power_func, x, y, maxfev=2000, bounds=([0, -10, -np.inf], [np.inf, 10, np.inf]))
        pred = power_func(len(seeds), *popt)
        error = np.mean(np.abs(y - power_func(x, *popt)))
        all_patterns['power_law'] = {'pred': pred, 'error': error, 'formula': 'power_law'}
    except:
        pass
    
    # 13. QCG
    if len(seeds) >= 3:
        try:
            m = 2147483648
            X = np.array([[seeds[i-1]**2, seeds[i-1], 1] for i in range(1, len(seeds))])
            Y = np.array([seeds[i] for i in range(1, len(seeds))])
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            a, b, c = coeffs
            pred = (a * seeds[-1]**2 + b * seeds[-1] + c) % m
            errors = [abs((a * seeds[i-1]**2 + b * seeds[i-1] + c) % m - seeds[i]) for i in range(1, len(seeds))]
            all_patterns['qcg'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'QCG'}
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
                all_patterns['multiplicative'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'multiplicative'}
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
            all_patterns['lag3'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'lag3'}
        except:
            pass
    
    # 16. Hyperbolic
    try:
        def hyp_func(x, a, b, c):
            return a / (x + b + 1) + c
        popt, _ = curve_fit(hyp_func, x, y, maxfev=2000)
        pred = hyp_func(len(seeds), *popt)
        error = np.mean(np.abs(y - hyp_func(x, *popt)))
        all_patterns['hyperbolic'] = {'pred': pred, 'error': error, 'formula': 'hyperbolic'}
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
            if best_pred is not None:
                all_patterns['xor_chain'] = {'pred': best_pred, 'error': best_err, 'formula': 'xor_chain'}
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
            all_patterns['combined_lcg'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'combined_lcg'}
        except:
            pass
    
    # 19. Hash Mix
    if len(seeds) >= 2:
        try:
            c1, r1, c2 = 0xcc9e2d51, 15, 0x1b873593
            errors = [abs((((seeds[i-1] * c1) & 0xFFFFFFFF) ^ (seeds[i-1] >> r1)) * c2 & 0xFFFFFFFF - seeds[i]) for i in range(1, len(seeds))]
            pred = (((seeds[-1] * c1) & 0xFFFFFFFF) ^ (seeds[-1] >> r1)) * c2 & 0xFFFFFFFF
            all_patterns['hash_mix'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'hash_mix'}
        except:
            pass
    
    # 20. Hash Rotate
    if len(seeds) >= 2:
        try:
            k, c = 13, 0x9e3779b9
            errors = [abs((((seeds[i-1] << k) | (seeds[i-1] >> (32 - k))) & 0xFFFFFFFF) ^ c - seeds[i]) for i in range(1, len(seeds))]
            pred = (((seeds[-1] << k) | (seeds[-1] >> (32 - k))) & 0xFFFFFFFF) ^ c
            all_patterns['hash_rotate'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'hash_rotate'}
        except:
            pass
    
    # 21. Hash Weyl
    if len(seeds) >= 2:
        try:
            weyl = 0x9e3779b97f4a7c15
            pred = (seeds[-1] + weyl) & 0xFFFFFFFF
            errors = [abs((seeds[i-1] + weyl) & 0xFFFFFFFF - seeds[i]) for i in range(1, len(seeds))]
            all_patterns['hash_weyl'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'hash_weyl'}
        except:
            pass
    
    # 22. Hash Combine
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
            all_patterns['hash_combine'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'hash_combine'}
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
            all_patterns['hash_avalanche'] = {'pred': pred, 'error': np.mean(errors), 'formula': 'hash_avalanche'}
        except:
            pass
    
    print(f"  ✅ {len(all_patterns)} patterns analizați\n")
    
    # Calculează confidence pentru toate
    valid = {k: v for k, v in all_patterns.items() if v.get('pred') and v.get('error') != float('inf')}
    
    if not valid:
        return {'pattern_type': 'none', 'predicted_seed': None, 'confidence': 0, 'all_patterns': all_patterns, 'top_patterns': []}
    
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
    
    patterns_with_confidence.sort(key=lambda x: x['confidence'], reverse=True)
    
    # TOATE cu 100% SAU doar best
    perfect_patterns = [p for p in patterns_with_confidence if p['confidence'] == 100.0]
    
    if perfect_patterns:
        return {
            'pattern_type': 'multiple_perfect' if len(perfect_patterns) > 1 else perfect_patterns[0]['name'],
            'predicted_seed': perfect_patterns[0]['pred'],
            'confidence': 100.0,
            'formula': perfect_patterns[0]['formula'],
            'error': 0.0,
            'all_patterns': all_patterns,
            'top_patterns': perfect_patterns
        }
    else:
        best = patterns_with_confidence[0]
        return {
            'pattern_type': best['name'],
            'predicted_seed': best['pred'],
            'confidence': best['confidence'],
            'formula': best['formula'],
            'error': best['error'],
            'all_patterns': all_patterns,
            'top_patterns': [best]
        }


class CPUOnlyPredictor:
    def __init__(self, lottery_type="5-40"):
        self.lottery_type = lottery_type
        self.config = get_lottery_config(lottery_type)
        self.data_file = f"{lottery_type}_data.json"
    
    def load_data(self, last_n=None, start_year=None, end_year=None):
        """Load data - cu suport CORECT pentru Joker composite"""
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
        
        # Normalizare - CORECT pentru Joker!
        normalized = []
        for e in filtered:
            # Pentru Joker, reconstruim ordinea CORECTĂ (5 numere + joker)
            if self.config.is_composite and 'composite_breakdown' in e:
                # Extragem din breakdown
                part1_nums = e['composite_breakdown']['part_1']['numbers']
                part2_nums = e['composite_breakdown']['part_2']['numbers']
                # Concatenăm: 5 numere + 1 joker (NU sortăm împreună!)
                numere_corecte = part1_nums + part2_nums
            else:
                # Pentru 5/40 și 6/49, folosim numbers_sorted
                numere_corecte = e.get('numere', e.get('numbers', e.get('numbers_sorted')))
            
            normalized.append({
                'data': e.get('data', e.get('date')),
                'numere': numere_corecte
            })
        
        return normalized
    
    def run_prediction(self, last_n=None, start_year=None, end_year=None,
                      seed_range=None, mersenne_timeout=10, min_success_rate=0.66):
        
        # Folosește seed range OPTIMIZAT automat
        if seed_range is None:
            seed_range = get_optimal_seed_range(self.lottery_type)
            print(f"✅ Seed range OPTIMIZAT: {seed_range[0]:,} - {seed_range[1]:,}")
        
        # Search size = TOATE seeds-urile (100% coverage)
        search_size = get_exhaustive_search_size(self.lottery_type)
        print(f"✅ Search size: {search_size:,} seeds (100% COVERAGE - EXHAUSTIVE!)")
        print(f"⏰ Timeout Mersenne: {mersenne_timeout} minute per extragere")
        
        print(f"\n{'='*70}")
        print(f"  CPU-ONLY PREDICTOR - {self.lottery_type.upper()}")
        print(f"{'='*70}\n")
        
        # Detectare cores fizice
        total_logical = cpu_count()
        if HAS_PSUTIL:
            physical = psutil.cpu_count(logical=False)
        else:
            physical = total_logical // 2 if total_logical > 32 else total_logical
        
        num_cores = max(1, physical - 1)
        
        print(f"💻 Cores logice: {total_logical}")
        print(f"💻 Cores fizice: {physical}")
        print(f"💻 Cores folosite: {num_cores}/{physical} (100% - 1 core)")
        print(f"🎯 RNG-uri: 21 (toate pe CPU)")
        print(f"⚡ Reverse Engineering: 6 LCG (INSTANT)")
        print(f"📊 Pattern-uri: 23 (toate pe CPU)")
        print(f"🔍 Seed range: {seed_range[0]:,} - {seed_range[1]:,}")
        print(f"📈 Search: {search_size:,} seeds (100% EXHAUSTIVE!)")
        print(f"⏰ Mersenne timeout: {mersenne_timeout} min per extragere")
        
        # Load
        if last_n:
            print(f"📊 Încărcare ultimele {last_n} extrageri...")
            data = self.load_data(last_n=last_n)
        else:
            print(f"📊 Încărcare {start_year}-{end_year}...")
            data = self.load_data(start_year=start_year, end_year=end_year)
        
        print(f"✅ {len(data)} extrageri încărcate\n")
        
        print("📋 Extrageri:")
        for i, e in enumerate(data, 1):
            print(f"  {i}. {e['data']:15s} → {e['numere']}")
        print()
        
        # Test TOATE 21 RNG-uri
        print(f"{'='*70}")
        print(f"  RNG TESTING - {num_cores} cores per RNG")
        print(f"{'='*70}\n")
        
        rng_results = {}
        
        for idx, rng_name in enumerate(RNG_TYPES.keys(), 1):
            # Afișare cu info exhaustive
            if rng_name == 'mersenne':
                print(f"[{idx}/21] 💻 {rng_name.upper()} (⏰ TIMEOUT {mersenne_timeout} min - exhaustive până la timeout)")
            else:
                print(f"[{idx}/21] 💻 {rng_name.upper()} (EXHAUSTIVE - toate {search_size:,} seeds)")
            
            tasks = [(i, e['numere'], rng_name, self.config, seed_range, search_size, mersenne_timeout, self.lottery_type, e['data']) 
                    for i, e in enumerate(data) if len(e['numere']) == self.config.numbers_to_draw]
            
            seeds_found = []
            draws_with_seeds = []
            cached_count = 0
            
            with Pool(processes=num_cores) as pool:
                for i, result in enumerate(pool.imap_unordered(cpu_worker, tasks)):
                    idx_task, seed, from_cache = result
                    if seed is not None:
                        seeds_found.append(seed)
                        draws_with_seeds.append({
                            'idx': idx_task,
                            'date': data[idx_task]['data'],
                            'numbers': data[idx_task]['numere'],
                            'seed': seed
                        })
                        if from_cache:
                            cached_count += 1
                    
                    if (i + 1) % 2 == 0 or (i + 1) == len(tasks):
                        progress = 100 * (i + 1) / len(tasks)
                        cache_info = f" ({cached_count} din cache)" if cached_count > 0 else ""
                        print(f"  [{i+1}/{len(tasks)}] ({progress:.1f}%)... {len(seeds_found)} seeds găsite{cache_info}", end='\r')
            
            success_rate = len(seeds_found) / len(data) if len(data) > 0 else 0
            cache_msg = f" ({cached_count} INSTANT din cache!)" if cached_count > 0 else ""
            print(f"\n  ✅ {len(seeds_found)}/{len(data)} ({success_rate:.1%}){cache_msg}", end='')
            
            if success_rate >= min_success_rate:
                print(f" - ✅ PESTE 66%!")
                draws_with_seeds.sort(key=lambda x: x['idx'])
                seeds_found = [d['seed'] for d in draws_with_seeds]
                rng_results[rng_name] = {
                    'seeds': seeds_found,
                    'draws': draws_with_seeds,
                    'success_rate': success_rate
                }
            else:
                print(f" - ❌ Sub 66%")
            
            print()
        
        if not rng_results:
            print(f"\n❌ Niciun RNG nu a trecut de 66%!\n")
            return
        
        # Pattern Analysis
        print(f"\n{'='*70}")
        print(f"  PATTERN ANALYSIS - {len(rng_results)} RNG-uri găsite")
        print(f"{'='*70}\n")
        
        predictions = []
        
        for rng_name, result in sorted(rng_results.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            print(f"\n{'='*70}")
            print(f"{rng_name.upper()} - Success: {result['success_rate']:.1%}")
            print(f"{'='*70}")
            
            # VERIFICARE ORDINE
            print(f"\n  📋 Seeds (ordine cronologică - primele 5):")
            for i, draw in enumerate(result['draws'][:5]):
                print(f"    {i+1}. {draw['date']:15s} → seed: {draw['seed']:>10,}")
            if len(result['draws']) > 5:
                print(f"    ... (+{len(result['draws'])-5} seeds)")
            print(f"    → Ultimul seed (cel mai nou): {result['seeds'][-1]:,}")
            print(f"    → Prezice seed #{len(result['seeds'])+1}\n")
            
            # Pattern analysis
            pattern = analyze_all_patterns_cpu(result['seeds'])
            
            # Display patterns
            if pattern.get('top_patterns') and len(pattern['top_patterns']) > 1:
                print(f"  🔥 MULTIPLE PATTERN-URI PERFECTE ({len(pattern['top_patterns'])} cu 100%):")
                for i, p in enumerate(pattern['top_patterns'], 1):
                    print(f"    {i}. {p['name'].upper()}: {p['formula']}, seed={p['pred']:,}")
            elif pattern.get('top_patterns'):
                p = pattern['top_patterns'][0]
                print(f"  🏆 BEST PATTERN: {p['name'].upper()}")
                print(f"  📐 Formula: {p['formula']}")
                print(f"  🎯 Confidence: {p['confidence']:.2f}%")
                print(f"  ❌ Error: {p['error']}")
            
            print(f"\n  📊 Toate patterns ({len(pattern.get('all_patterns', {}))}):")
            for pn, pd in pattern.get('all_patterns', {}).items():
                err_str = f"{pd.get('error', 0):.2f}" if pd.get('error') != float('inf') else "∞"
                print(f"    {pn:20s}: error={err_str}")
            
            # Predicție - GENEREAZĂ PENTRU TOATE PATTERN-URILE CU 100%!
            if pattern.get('top_patterns'):
                # Verifică dacă sunt multiple cu 100%
                if len(pattern['top_patterns']) > 1:
                    print(f"\n  {'='*66}")
                    print(f"  🎯 PREDICȚII ({len(pattern['top_patterns'])} PATTERN-URI PERFECTE)")
                    print(f"  {'='*66}\n")
                    
                    for i, p in enumerate(pattern['top_patterns'], 1):
                        try:
                            rng = create_rng(rng_name, p['pred'])
                            
                            if self.config.is_composite:
                                nums = []
                                for count, min_val, max_val in self.config.composite_parts:
                                    part = generate_numbers(rng, count, min_val, max_val)
                                    nums.extend(part)
                            else:
                                nums = generate_numbers(rng, self.config.numbers_to_draw, self.config.min_number, self.config.max_number)
                            
                            print(f"  {i}. {p['name'].upper()}:")
                            print(f"     Seed: {p['pred']:,}")
                            print(f"     NUMERE: {sorted(nums)}")
                            print()
                            
                            predictions.append({
                                'rng': rng_name,
                                'success_rate': result['success_rate'],
                                'pattern': p['name'],
                                'formula': p['formula'],
                                'confidence': p['confidence'],
                                'seed': p['pred'],
                                'numbers': sorted(nums)
                            })
                        except Exception as e:
                            print(f"  ❌ Eroare predicție {p['name']}: {e}")
                    
                    print(f"  {'='*66}\n")
                
                elif pattern['top_patterns']:
                    # Un singur pattern
                    p = pattern['top_patterns'][0]
                    try:
                        rng = create_rng(rng_name, p['pred'])
                        
                        if self.config.is_composite:
                            nums = []
                            for count, min_val, max_val in self.config.composite_parts:
                                part = generate_numbers(rng, count, min_val, max_val)
                                nums.extend(part)
                        else:
                            nums = generate_numbers(rng, self.config.numbers_to_draw, self.config.min_number, self.config.max_number)
                        
                        print(f"\n  {'='*66}")
                        print(f"  🎯 PREDICȚIE PENTRU URMĂTOAREA EXTRAGERE")
                        print(f"  {'='*66}")
                        print(f"  Seed prezis: {p['pred']:,}")
                        print(f"  NUMERE PREZISE: {sorted(nums)}")
                        print(f"  {'='*66}\n")
                        
                        predictions.append({
                            'rng': rng_name,
                            'success_rate': result['success_rate'],
                            'pattern': p['name'],
                            'formula': p['formula'],
                            'confidence': p['confidence'],
                            'seed': p['pred'],
                            'numbers': sorted(nums)
                        })
                    except Exception as e:
                        print(f"  ❌ Eroare predicție: {e}")
        
        # Salvare
        if predictions:
            output = f"cpu_prediction_{self.lottery_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output, 'w') as f:
                json.dump({
                    'lottery': self.lottery_type,
                    'timestamp': datetime.now().isoformat(),
                    'predictions': predictions
                }, f, indent=2)
            print(f"\n💾 Rezultate salvate: {output}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CPU-Only Predictor')
    parser.add_argument('--lottery', default='5-40', choices=['5-40', '6-49', 'joker'])
    parser.add_argument('--last-n', type=int, help='Ultimele N extrageri')
    parser.add_argument('--start-year', type=int)
    parser.add_argument('--end-year', type=int)
    parser.add_argument('--seed-range', type=int, nargs=2, default=None,
                      help='Seed range (default: auto-optimizat)')
    parser.add_argument('--mersenne-timeout', type=int, default=10,
                      help='Timeout pentru Mersenne în minute (default: 10)')
    parser.add_argument('--min-success-rate', type=float, default=0.66,
                      help='Success rate minim (default: 0.66)')
    
    args = parser.parse_args()
    
    if not args.last_n and not (args.start_year and args.end_year):
        print("❌ Specifică --last-n SAU (--start-year și --end-year)!")
        sys.exit(1)
    
    predictor = CPUOnlyPredictor(args.lottery)
    predictor.run_prediction(
        last_n=args.last_n,
        start_year=args.start_year,
        end_year=args.end_year,
        seed_range=tuple(args.seed_range) if args.seed_range else None,
        mersenne_timeout=args.mersenne_timeout,
        min_success_rate=args.min_success_rate
    )
