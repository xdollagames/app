#!/usr/bin/env python3
"""
CPU-ONLY PREDICTOR - ZERO GPU
Simplu, stabil, rapid cu reverse engineering
OPTIMIZAT: Chunk size mare + Numba JIT pentru SIMD
"""

import json
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
import random

try:
    from numba import njit
    HAS_NUMBA = True
except:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        """Fallback dacă numba nu e disponibil"""
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

try:
    import psutil
    HAS_PSUTIL = True
except:
    HAS_PSUTIL = False

from lottery_config import get_lottery_config
from advanced_rng_library import RNG_TYPES, create_rng, generate_numbers


# OPTIMIZARE SIMD: Funcții vectorizate pentru LCG-uri (cele mai rapide)
@njit(cache=True)
def lcg_batch(seeds, a, c, m, min_val, max_val, count):
    """Procesează batch de seeds LCG simultan cu SIMD/vectorizare"""
    range_size = max_val - min_val + 1
    batch_size = len(seeds)
    results = np.zeros((batch_size, count), dtype=np.int32)
    
    for b in range(batch_size):
        state = seeds[b] % m
        numbers = []
        seen = set()
        attempts = 0
        max_attempts = count * 100
        
        while len(numbers) < count and attempts < max_attempts:
            # LCG step
            state = (a * state + c) % m
            num = min_val + (state % range_size)
            
            if num not in seen:
                numbers.append(num)
                seen.add(num)
            attempts += 1
        
        # Umple rezultatul
        for i in range(min(len(numbers), count)):
            results[b, i] = numbers[i]
    
    return results


if HAS_NUMBA:
    print("✅ Numba SIMD optimization activată!")
else:
    print("⚠️  Numba nu e disponibil - rulează fără SIMD")



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
    """Verifică cache - returnează seed SAU 'NOT_FOUND' SAU None"""
    cache = load_seeds_cache()
    result = cache.get(lottery_type, {}).get(date_str, {}).get(rng_name)
    return result  # Poate fi: seed (int), 'NOT_FOUND' (str), sau None

def cache_seed(lottery_type, date_str, rng_name, seed):
    """Salvează seed în cache (seed=int SAU seed='NOT_FOUND')"""
    cache = load_seeds_cache()
    
    if lottery_type not in cache:
        cache[lottery_type] = {}
    if date_str not in cache[lottery_type]:
        cache[lottery_type][date_str] = {}
    
    cache[lottery_type][date_str][rng_name] = seed
    save_seeds_cache(cache)


# SEED RANGES pentru fiecare RNG - TOATE DISPONIBILE (fără Mersenne)
ALL_RNGS = {
    # LCG family (6 RNG-uri)
    'lcg_borland': 2**32,
    'lcg_glibc': 2**31,
    'lcg_minstd': 2**31 - 1,
    'lcg_randu': 2**31,
    'lcg_weak': 233280,
    'php_rand': 2**31 - 1,
    
    # Xorshift family (4 RNG-uri)
    'xorshift32': 2**32 - 1,
    'xorshift64': 2**32,        # Poate merge la 2^64 pentru 6-49
    'xorshift128': 2**32,       # Poate merge la 2^64 pentru 6-49
    'xorshift_simple': 2**32,
    
    # Modern RNGs (3 RNG-uri)
    'pcg32': 2**32,
    'xoshiro256': 2**32,        # Poate merge la 2^64 pentru 6-49
    'splitmix': 2**32,          # Poate merge la 2^64 pentru 6-49
    
    # Other algorithms (4 RNG-uri)
    'mwc': 2**32,
    'fibonacci': 2**31,
    'middlesquare': 2**32,
    'lfsr': 2**32,
    
    # Language-specific (3 RNG-uri)
    'js_math_random': 2**32,
    'java_random': 2**32,
    'complex_hash': 2**32,
    
    # NEW - Algoritmi conceptual diferiți (4 RNG-uri)
    'lxm': 2**32,               # LCG + Xorshift hybrid
    'xoroshiro128': 2**64,      # Xorshift cu rotații + 64-bit
    'chacha': 2**32,            # ChaCha-based
    'kiss': 2**32,              # KISS - combinație 3 RNG-uri
    
    # NEW - 64-bit specifice pentru 6-49 (2 RNG-uri)
    'pcg64': 2**64,             # PCG 64-bit - top quality
    'lxm64': 2**64,             # LXM 64-bit - robust modern
}
# TOTAL: 26 RNG-uri

# RNG-uri care pot scala la 64-bit DOAR pentru 6-49
RNG_64BIT_CAPABLE = ['xorshift64', 'xorshift128', 'xoshiro256', 'splitmix', 'xoroshiro128', 'pcg64', 'lxm64']

# Calcul factorial
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Calcul combinații C(n, k)
def comb(n, k):
    if k > n:
        return 0
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

# POSIBILITĂȚI TOTALE per joc (cu ORDINE EXACTĂ!)
LOTTERY_POSSIBILITIES = {
    '5-40': comb(40, 6) * factorial(6),           # C(40,6) × 6! = 2,763,633,600
    '6-49': comb(49, 6) * factorial(6),           # C(49,6) × 6! = 10,068,347,520
    'joker': comb(45, 5) * factorial(5) * 20,     # C(45,5) × 5! × 20 = 2,932,221,600
}

def get_rng_max_seeds(rng_name, lottery_type='5-40'):
    """Returnează numărul MAXIM de seeds pentru un RNG specific și loterie"""
    # Pentru 6-49, folosește 64-bit pentru RNG-urile capabile
    if lottery_type == '6-49' and rng_name in RNG_64BIT_CAPABLE:
        return 2**64
    # Altfel, folosește range-ul standard
    return ALL_RNGS.get(rng_name, 2**32)

def get_compatible_rngs(lottery_type):
    """Returnează TOATE RNG-urile care au range suficient pentru loteria specificată"""
    min_required = LOTTERY_POSSIBILITIES.get(lottery_type, 0)
    compatible = []
    
    for rng_name in ALL_RNGS.keys():
        max_seeds = get_rng_max_seeds(rng_name, lottery_type)
        if max_seeds >= min_required:
            compatible.append(rng_name)
    
    return compatible


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
                # CRITIC: Comparăm ORDINEA EXACTĂ de extragere!
                if generated == numbers:
                    return seed
            except:
                pass
    
    return None


def cpu_worker_chunked(args):
    """Worker CPU - OPTIMIZAT: testează fiecare seed o singură dată pentru TOATE extragerile"""
    import time
    
    all_targets, rng_name, lottery_config, seed_chunk_start, seed_chunk_end, timeout_seconds, lottery_type, seed_range_tuple = args
    # all_targets = lista cu toate extragerile: [(idx, numbers, date), ...]
    
    start_time = time.time()
    results_per_draw = {idx: None for idx, _, _ in all_targets}
    
    # Verifică cache pentru fiecare extragere
    cache_hits = {}
    for idx, numbers, date_str in all_targets:
        cached_result = get_cached_seed(lottery_type, date_str, rng_name)
        
        if cached_result == 'NOT_FOUND':
            results_per_draw[idx] = 'NOT_FOUND'
        elif cached_result is not None and isinstance(cached_result, int):
            if seed_chunk_start <= cached_result < seed_chunk_end:
                cache_hits[idx] = (cached_result, date_str)
    
    # Verifică cache hits
    for idx, (cached_seed, date_str) in cache_hits.items():
        try:
            rng = create_rng(rng_name, cached_seed)
            if lottery_config.is_composite:
                generated = []
                count_1, min_1, max_1 = lottery_config.composite_parts[0]
                part_1 = generate_numbers(rng, count_1, min_1, max_1)
                generated.extend(part_1)
                count_2, min_2, max_2 = lottery_config.composite_parts[1]
                joker = min_2 + (rng.next() % (max_2 - min_2 + 1))
                generated.append(joker)
            else:
                generated = generate_numbers(rng, lottery_config.numbers_to_draw, lottery_config.min_number, lottery_config.max_number)
            
            # Compară cu target pentru această extragere
            target = [num for i, num, _ in all_targets if i == idx][0]
            if generated == target:
                results_per_draw[idx] = cached_seed
        except:
            pass
    
    # Reverse engineering pentru primul chunk (doar pentru extragerea 0)
    if seed_chunk_start == 0 and all_targets:
        idx_0, numbers_0, date_0 = all_targets[0]
        if results_per_draw[idx_0] is None:
            reversed_seed = try_reverse_engineering(rng_name, numbers_0, lottery_config)
            if reversed_seed is not None:
                cache_seed(lottery_type, date_0, rng_name, reversed_seed)
                results_per_draw[idx_0] = reversed_seed
    
    # LOOP OPTIMIZAT: testează fiecare seed o singură dată
    for seed in range(seed_chunk_start, seed_chunk_end):
        if (time.time() - start_time) > timeout_seconds:
            break
        
        try:
            rng = create_rng(rng_name, seed)
            
            # Generează ODATĂ
            if lottery_config.is_composite:
                generated = []
                count_1, min_1, max_1 = lottery_config.composite_parts[0]
                part_1 = generate_numbers(rng, count_1, min_1, max_1)
                generated.extend(part_1)
                count_2, min_2, max_2 = lottery_config.composite_parts[1]
                joker = min_2 + (rng.next() % (max_2 - min_2 + 1))
                generated.append(joker)
            else:
                generated = generate_numbers(rng, lottery_config.numbers_to_draw, lottery_config.min_number, lottery_config.max_number)
            
            # Compară cu TOATE extragerile
            for idx, target, date_str in all_targets:
                if results_per_draw[idx] is None:  # Doar dacă nu am găsit deja
                    if generated == target:
                        cache_seed(lottery_type, date_str, rng_name, seed)
                        results_per_draw[idx] = seed
                        # Nu break - poate să matchuiască și alte extrageri!
        except:
            continue
    
    # Marchează NOT_FOUND pentru ultimul chunk
    if seed_chunk_end >= seed_range_tuple[1]:
        for idx, numbers, date_str in all_targets:
            if results_per_draw[idx] is None:
                cache_seed(lottery_type, date_str, rng_name, 'NOT_FOUND')
                results_per_draw[idx] = 'NOT_FOUND'
    
    # Returnează rezultate pentru toate extragerile
    return results_per_draw
    """Worker CPU - cu CACHE pentru seeds deja găsite!"""
    import time
    
    draw_idx, numbers, rng_name, lottery_config, seed_range, search_size_total, timeout_seconds, lottery_type, date_str = args
    # CRITIC: Căutăm seed-ul care reproduce ORDINEA EXACTĂ de extragere!
    target_exact = numbers
    start_time = time.time()
    
    # VERIFICĂ CACHE MAI ÎNTÂI! (INSTANT dacă există!)
    cached_seed = get_cached_seed(lottery_type, date_str, rng_name)
    if cached_seed is not None:
        # Verifică că seed-ul e încă valid
        try:
            rng = create_rng(rng_name, cached_seed)
            if lottery_config.is_composite:
                # FIX JOKER: Permite duplicate! (13.7% din cazuri în date reale)
                generated = []
                
                # Partea 1: Generează primele 5 numere UNIQUE din range-ul lor
                count_1, min_1, max_1 = lottery_config.composite_parts[0]
                part_1 = generate_numbers(rng, count_1, min_1, max_1)
                generated.extend(part_1)
                
                # Partea 2: Generează Joker FĂRĂ verificare duplicate!
                count_2, min_2, max_2 = lottery_config.composite_parts[1]
                # Generează direct UN număr (permite duplicate!)
                joker = min_2 + (rng.next() % (max_2 - min_2 + 1))
                generated.append(joker)
            else:
                generated = generate_numbers(rng, lottery_config.numbers_to_draw, lottery_config.min_number, lottery_config.max_number)
            
            # Comparăm ORDINEA EXACTĂ de extragere
            if generated == target_exact:
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
            
            # Composite support (Joker) - FIX JOKER: Permite duplicate!
            if lottery_config.is_composite:
                # FIX JOKER: Permite duplicate! (13.7% din cazuri în date reale)
                generated = []
                
                # Partea 1: Generează primele 5 numere UNIQUE din range-ul lor
                count_1, min_1, max_1 = lottery_config.composite_parts[0]
                part_1 = generate_numbers(rng, count_1, min_1, max_1)
                generated.extend(part_1)
                
                # Partea 2: Generează Joker FĂRĂ verificare duplicate!
                count_2, min_2, max_2 = lottery_config.composite_parts[1]
                # Generează direct UN număr (permite duplicate!)
                joker = min_2 + (rng.next() % (max_2 - min_2 + 1))
                generated.append(joker)
            else:
                generated = generate_numbers(rng, lottery_config.numbers_to_draw, lottery_config.min_number, lottery_config.max_number)
            
            # Comparăm ORDINEA EXACTĂ de extragere
            if generated == target_exact:
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
                      rng_timeout_minutes=60, min_success_rate=0.66, min_recent_seeds=None):
        
        print(f"\n{'='*70}")
        print(f"  CPU-ONLY PREDICTOR - {self.lottery_type.upper()}")
        print(f"  ORDINEA EXACTĂ - RANGE MAXIM PER RNG")
        print(f"{'='*70}\n")
        
        # Detectare cores - FOLOSEȘTE TOATE!
        total_logical = cpu_count()
        num_cores = total_logical  # TOATE threadurile
        
        # Filtrează RNG-urile compatibile cu acest joc
        rng_list = get_compatible_rngs(self.lottery_type)
        total_possibilities = LOTTERY_POSSIBILITIES.get(self.lottery_type, 0)
        
        print(f"💻 Cores logice: {total_logical}")
        print(f"💻 Cores folosite: {num_cores} (TOATE!)")
        print(f"🎲 Posibilități totale {self.lottery_type}: {total_possibilities:,}")
        print(f"🎯 RNG-uri compatibile: {len(rng_list)}")
        print(f"⚡ Reverse Engineering: 6 LCG (INSTANT)")
        if HAS_NUMBA:
            print(f"🚀 Optimizare: Numba JIT + SIMD activată")
        print(f"⏰ Timeout GLOBAL per RNG: {rng_timeout_minutes} minute")
        print(f"🔍 Comparare: ORDINEA EXACTĂ (nu sorted!)")
        
        # Load
        if last_n:
            print(f"\n📊 Încărcare ultimele {last_n} extrageri...")
            data = self.load_data(last_n=last_n)
        else:
            print(f"\n📊 Încărcare {start_year}-{end_year}...")
            data = self.load_data(start_year=start_year, end_year=end_year)
        
        print(f"✅ {len(data)} extrageri încărcate\n")
        
        print("📋 Extrageri:")
        for i, e in enumerate(data, 1):
            print(f"  {i}. {e['data']:15s} → {e['numere']}")
        print()
        
        # Test RNG-urile selectate
        print(f"\n{'='*70}")
        print(f"  RNG TESTING - {len(rng_list)} RNG-uri")
        print(f"  Timeout: {rng_timeout_minutes} min per RNG")
        print(f"{'='*70}\n")
        
        rng_results = {}
        
        for idx, rng_name in enumerate(rng_list, 1):
            # Range pentru acest RNG și loterie
            max_seeds = get_rng_max_seeds(rng_name, self.lottery_type)
            seed_range = (0, max_seeds)
            
            # Timeout în secunde
            timeout_seconds = rng_timeout_minutes * 60
            
            print(f"[{idx}/{len(rng_list)}] 💻 {rng_name.upper()}")
            print(f"  📊 Range: {seed_range[0]:,} - {seed_range[1]:,} ({seed_range[1]:,} seeds)")
            print(f"  ⏰ Timeout: {rng_timeout_minutes} minute ({timeout_seconds} secunde)")
            
            # Creează CHUNKS OPTIMIZATE pentru RAM
            # Chunk mai mare = mai puțin overhead, folosește RAM pentru caching
            chunk_size = max(500000, seed_range[1] // (num_cores * 4))  # 4 chunks per core (era 10)
            
            # OPTIMIZARE: Creează tasks ODATĂ pentru TOATE extragerile
            # În loc să creeze tasks separate per extragere (ineficient)
            all_targets = [(i, e['numere'], e['data']) for i, e in enumerate(data) 
                          if len(e['numere']) == self.config.numbers_to_draw]
            
            tasks = []
            for chunk_start in range(seed_range[0], seed_range[1], chunk_size):
                chunk_end = min(chunk_start + chunk_size, seed_range[1])
                # Un task procesează un chunk pentru TOATE extragerile
                tasks.append((all_targets, rng_name, self.config, chunk_start, chunk_end, 
                            timeout_seconds, self.lottery_type, seed_range))
            
            print(f"  🔥 {len(tasks)} task-uri (chunks de {chunk_size:,}) → {min(num_cores, len(tasks))} cores active")
            print(f"  ⚡ OPTIMIZAT: Fiecare seed testat o singură dată pentru {len(all_targets)} extrageri!")
            
            seeds_found = []
            draws_with_seeds = []
            cached_positive = 0
            cached_negative = 0
            seeds_by_draw = {}
            seeds_processed = 0  # Număr total de seeds procesate
            total_seeds = seed_range[1]  # Total seeds pentru acest RNG
            
            rng_start_time = time.time()
            last_update_time = time.time()
            update_interval = 30.0  # Actualizare la fiecare 30 secunde
            
            with Pool(processes=num_cores) as pool:
                tasks_completed = 0
                for chunk_results in pool.imap_unordered(cpu_worker_chunked, tasks):
                    tasks_completed += 1
                    
                    # Incrementăm seeds procesate (fiecare task = un chunk)
                    seeds_processed += chunk_size
                    if seeds_processed > total_seeds:
                        seeds_processed = total_seeds
                    
                    # Check timeout global per RNG
                    elapsed = time.time() - rng_start_time
                    if elapsed > timeout_seconds:
                        print(f"\n  ⏰ TIMEOUT reached ({rng_timeout_minutes} min) - stopping RNG {rng_name}")
                        pool.terminate()
                        pool.join()
                        break
                    
                    # Procesează rezultatele pentru TOATE extragerile din acest chunk
                    seed_found_in_chunk = False
                    for idx_task, result_seed in chunk_results.items():
                        if result_seed == 'NOT_FOUND':
                            cached_negative += 1
                            seeds_by_draw[idx_task] = None
                        elif result_seed is not None and isinstance(result_seed, int) and idx_task not in seeds_by_draw:
                            seeds_by_draw[idx_task] = result_seed
                            seeds_found.append(result_seed)
                            draws_with_seeds.append({
                                'idx': idx_task,
                                'date': data[idx_task]['data'],
                                'numbers': data[idx_task]['numere'],
                                'seed': result_seed
                            })
                            seed_found_in_chunk = True
                            
                            # Afișăm imediat când găsim
                            print(f"\n  🎯 GĂSIT! Seed {result_seed:,} pentru {data[idx_task]['data']}: {data[idx_task]['numere']}")
                    
                    if seed_found_in_chunk:
                        last_update_time = time.time()  # Forțează update imediat
                        cached_positive += len([s for s in chunk_results.values() if isinstance(s, int)])
                    
                    # EARLY STOPPING: Dacă am găsit toate extragerile, STOP!
                    if len(seeds_found) == len(data):
                        print(f"\n  ✅ TOATE EXTRAGERILE GĂSITE! Opresc căutarea pentru {rng_name}")
                        pool.terminate()
                        pool.join()
                        break
                    
                    # Calculăm seeds_progress pentru verificare
                    seeds_progress = 100 * seeds_processed / total_seeds
                    
                    # STOP la 100% seeds procesate
                    if seeds_progress >= 100:
                        print(f"\n  ✅ 100% seeds procesate pentru {rng_name}")
                        pool.terminate()
                        pool.join()
                        break
                    
                    # UPDATE PROGRESS la fiecare 30 secunde SAU dacă e seed găsit
                    current_time = time.time()
                    if current_time - last_update_time >= update_interval or seed is not None:
                        last_update_time = current_time
                        
                        completed = len(seeds_by_draw)
                        progress = 100 * completed / len(data) if len(data) > 0 else 0
                        elapsed_min = elapsed / 60
                        
                        # Format seeds processed (în milioane, miliarde, etc)
                        if seeds_processed >= 1_000_000_000:
                            seeds_str = f"{seeds_processed/1_000_000_000:.1f}B"
                        elif seeds_processed >= 1_000_000:
                            seeds_str = f"{seeds_processed/1_000_000:.0f}M"
                        else:
                            seeds_str = f"{seeds_processed:,}"
                        
                        if total_seeds >= 1_000_000_000:
                            total_str = f"{total_seeds/1_000_000_000:.1f}B"
                        elif total_seeds >= 1_000_000:
                            total_str = f"{total_seeds/1_000_000:.0f}M"
                        else:
                            total_str = f"{total_seeds:,}"
                        
                        cache_info = ""
                        if cached_positive > 0:
                            cache_info += f" ✅{cached_positive}"
                        if cached_negative > 0:
                            cache_info += f" ⏭️{cached_negative}"
                        
                        print(f"  [{completed}/{len(data)}] ({progress:.1f}%) | {len(seeds_found)} seeds | {elapsed_min:.1f}/{rng_timeout_minutes}min | Seeds: {seeds_progress:.1f}% ({seeds_str}/{total_str}){cache_info}", end='\r')
            
            elapsed_total = time.time() - rng_start_time
            success_rate = len(seeds_found) / len(data) if len(data) > 0 else 0
            
            print(f"\n  ⏱️  Timp: {elapsed_total/60:.1f} minute")
            print(f"  ✅ {len(seeds_found)}/{len(data)} ({success_rate:.1%})", end='')
            
            # Detectează cel mai lung combo consecutiv de la FINAL
            draws_with_seeds.sort(key=lambda x: x['idx'])
            found_indices = [d['idx'] for d in draws_with_seeds]
            
            # Găsește combo consecutiv de la final
            consecutive_combo = []
            if found_indices:
                # Începe de la ultimul index posibil (len(data)-1) și merge înapoi
                for i in range(len(data) - 1, -1, -1):
                    if i in found_indices:
                        consecutive_combo.insert(0, i)  # Adaugă la început
                    else:
                        break  # Primul gap oprește combo-ul
            
            # Verifică dacă avem predicție validă
            valid_for_prediction = False
            use_combo_only = False
            
            if min_recent_seeds:
                # Mod RECENT-SEEDS: trebuie ultimele N consecutive
                if len(consecutive_combo) >= min_recent_seeds:
                    valid_for_prediction = True
                    # Dacă avem mai multe seeds decât combo-ul, folosim AMBELE
                    if len(found_indices) > len(consecutive_combo):
                        use_combo_only = False  # Facem 2 predicții
                        print(f" - ✅ ULTIMELE {len(consecutive_combo)} consecutive + {len(found_indices)-len(consecutive_combo)} mai vechi")
                    else:
                        use_combo_only = True  # Doar 1 predicție (toate sunt consecutive)
                        print(f" - ✅ ULTIMELE {len(consecutive_combo)} consecutive găsite!")
                else:
                    print(f" - ❌ Nu are ultimele {min_recent_seeds} consecutive (doar {len(consecutive_combo)})")
            else:
                # Mod normal success rate
                if success_rate >= min_success_rate:
                    valid_for_prediction = True
                    print(f" - ✅ PESTE {min_success_rate:.0%}!")
                    # Verificăm dacă avem combo mai mic decât total
                    if len(consecutive_combo) >= 2 and len(found_indices) > len(consecutive_combo):
                        use_combo_only = False  # Facem 2 predicții
                    else:
                        use_combo_only = True
                else:
                    print(f" - ❌ Sub {min_success_rate:.0%}")
            
            # Salvează rezultatele dacă e valid
            if valid_for_prediction:
                all_seeds = [d['seed'] for d in draws_with_seeds]
                combo_draws = [d for d in draws_with_seeds if d['idx'] in consecutive_combo]
                combo_seeds = [d['seed'] for d in combo_draws]
                
                rng_results[rng_name] = {
                    'all_seeds': all_seeds,
                    'all_draws': draws_with_seeds,
                    'combo_seeds': combo_seeds,
                    'combo_draws': combo_draws,
                    'consecutive_count': len(consecutive_combo),
                    'total_count': len(found_indices),
                    'has_gaps': len(found_indices) > len(consecutive_combo),
                    'success_rate': success_rate
                }
            
            print()
        
        if not rng_results:
            if min_recent_seeds:
                print(f"\n❌ Niciun RNG nu are ultimele {min_recent_seeds} seeds consecutive!\n")
            else:
                print(f"\n❌ Niciun RNG nu a trecut de {min_success_rate:.0%}!\n")
            return
        
        # Pattern Analysis
        print(f"\n{'='*70}")
        print(f"  PATTERN ANALYSIS - {len(rng_results)} RNG-uri găsite")
        print(f"{'='*70}\n")
        
        predictions = []
        
        for rng_name, result in sorted(rng_results.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            print(f"\n{'='*70}")
            print(f"{rng_name.upper()} - Success: {result['success_rate']:.1%}", end='')
            if result['has_gaps']:
                print(f" (Total: {result['total_count']}, Consecutiv final: {result['consecutive_count']})", end='')
            print()
            print(f"{'='*70}")
            
            # Afișare seeds găsite
            print(f"\n  📋 Seeds găsite (ordine cronologică):")
            for i, draw in enumerate(result['all_draws'][:10]):
                marker = "🔥" if draw['idx'] in [d['idx'] for d in result['combo_draws']] else "  "
                print(f"    {marker} {i+1}. {draw['date']:15s} → seed: {draw['seed']:>10,}")
            if len(result['all_draws']) > 10:
                print(f"    ... (+{len(result['all_draws'])-10} seeds)")
            
            # Afișare combo consecutiv
            if result['has_gaps']:
                print(f"\n  🔥 COMBO CONSECUTIV (ultimele {result['consecutive_count']}):")
                for i, draw in enumerate(result['combo_draws']):
                    print(f"     {i+1}. {draw['date']:15s} → seed: {draw['seed']:>10,}")
            
            # PREDICȚIE 1: Cu TOATE seeds-urile (dacă are gaps)
            if result['has_gaps'] and len(result['all_seeds']) >= 2:
                print(f"\n  {'─'*66}")
                print(f"  PREDICȚIE #1: Pattern pe TOATE {len(result['all_seeds'])} seeds")
                print(f"  {'─'*66}")
                
                pattern_all = analyze_all_patterns_cpu(result['all_seeds'])
                
                if pattern_all and pattern_all.get('top_patterns'):
                    for p in pattern_all['top_patterns'][:1]:  # Top 1
                        print(f"  Pattern: {p['name']}")
                        print(f"  Confidence: {p['confidence']:.1f}%")
                        print(f"  Seed prezis (toate): {p['pred']:,}")
                        
                        # Generează predicție
                        try:
                            rng = create_rng(rng_name, p['pred'])
                            if self.config.is_composite:
                                nums = []
                                count_1, min_1, max_1 = self.config.composite_parts[0]
                                part_1 = generate_numbers(rng, count_1, min_1, max_1)
                                nums.extend(part_1)
                                count_2, min_2, max_2 = self.config.composite_parts[1]
                                joker = min_2 + (rng.next() % (max_2 - min_2 + 1))
                                nums.append(joker)
                            else:
                                nums = generate_numbers(rng, self.config.numbers_to_draw, self.config.min_number, self.config.max_number)
                            
                            print(f"  Predicție: {nums}")
                            predictions.append({
                                'rng': rng_name,
                                'type': 'ALL_SEEDS',
                                'seeds_used': len(result['all_seeds']),
                                'pattern': p['name'],
                                'confidence': p['confidence'],
                                'numbers': nums,
                                'priority': 'low'
                            })
                        except Exception as e:
                            print(f"  ❌ Eroare: {e}")
            
            # PREDICȚIE #2: Cu DOAR COMBO-ul consecutiv (PRIORITAR!)
            if len(result['combo_seeds']) >= 2:
                print(f"\n  {'─'*66}")
                print(f"  🔥 PREDICȚIE #2: Pattern pe COMBO consecutiv ({len(result['combo_seeds'])} seeds) - PRIORITAR!")
                print(f"  {'─'*66}")
                
                pattern_combo = analyze_all_patterns_cpu(result['combo_seeds'])
                
                if pattern_combo and pattern_combo.get('top_patterns'):
                    for p in pattern_combo['top_patterns'][:1]:
                        print(f"  Pattern: {p['name']}")
                        print(f"  Confidence: {p['confidence']:.1f}%")
                        print(f"  Seed prezis (combo): {p['pred']:,}")
                        
                        try:
                            rng = create_rng(rng_name, p['pred'])
                            if self.config.is_composite:
                                nums = []
                                count_1, min_1, max_1 = self.config.composite_parts[0]
                                part_1 = generate_numbers(rng, count_1, min_1, max_1)
                                nums.extend(part_1)
                                count_2, min_2, max_2 = self.config.composite_parts[1]
                                joker = min_2 + (rng.next() % (max_2 - min_2 + 1))
                                nums.append(joker)
                            else:
                                nums = generate_numbers(rng, self.config.numbers_to_draw, self.config.min_number, self.config.max_number)
                            
                            print(f"  🎯 Predicție PRIORITARĂ: {nums}")
                            predictions.append({
                                'rng': rng_name,
                                'type': 'CONSECUTIVE_COMBO',
                                'seeds_used': len(result['combo_seeds']),
                                'consecutive': True,
                                'pattern': p['name'],
                                'confidence': p['confidence'],
                                'numbers': nums,
                                'priority': 'HIGH'
                            })
                        except Exception as e:
                            print(f"  ❌ Eroare: {e}")
        
        # Salvare
        if predictions:
            # Sortăm: prioritate HIGH primele
            predictions.sort(key=lambda x: (x['priority'] != 'HIGH', -x['confidence']))
            
            output = f"cpu_prediction_{self.lottery_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output, 'w') as f:
                json.dump({
                    'lottery': self.lottery_type,
                    'timestamp': datetime.now().isoformat(),
                    'predictions': predictions
                }, f, indent=2)
            print(f"\n💾 Rezultate salvate: {output}")
            
            # Afișare rezumat predicții
            print(f"\n{'='*70}")
            print(f"  REZUMAT PREDICȚII")
            print(f"{'='*70}")
            
            high_priority = [p for p in predictions if p.get('priority') == 'HIGH']
            low_priority = [p for p in predictions if p.get('priority') == 'low']
            
            if high_priority:
                print(f"\n🔥 PREDICȚII PRIORITARE (combo consecutiv final):")
                for i, p in enumerate(high_priority[:5], 1):
                    print(f"  {i}. {p['rng'].upper()} - {p['seeds_used']} seeds consecutiv")
                    print(f"     Pattern: {p['pattern']} ({p['confidence']:.1f}%)")
                    print(f"     → {p['numbers']}")
            
            if low_priority:
                print(f"\n📊 Predicții suplimentare (include gap-uri în istoric):")
                for i, p in enumerate(low_priority[:5], 1):
                    print(f"  {i}. {p['rng'].upper()} - {p['seeds_used']} seeds total")
                    print(f"     Pattern: {p['pattern']} ({p['confidence']:.1f}%)")
                    print(f"     → {p['numbers']}")
            
            print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CPU-Only Predictor - ORDINEA EXACTĂ')
    parser.add_argument('--lottery', default='5-40', choices=['5-40', '6-49', 'joker'])
    parser.add_argument('--last-n', type=int, help='Ultimele N extrageri')
    parser.add_argument('--start-year', type=int)
    parser.add_argument('--end-year', type=int)
    parser.add_argument('--rng-timeout', type=int, default=60,
                      help='Timeout GLOBAL per RNG în minute (default: 60)')
    parser.add_argument('--min-success-rate', type=float, default=0.66,
                      help='Success rate minim (default: 0.66). Folosește 1.0 pentru doar RNG-uri PERFECTE (100%%)')
    parser.add_argument('--only-perfect', action='store_true',
                      help='Acceptă doar RNG-uri cu 100%% success rate (shortcut pentru --min-success-rate 1.0)')
    parser.add_argument('--min-recent-seeds', type=int, default=None,
                      help='Permite predicție dacă găsește ultimele N seeds (ex: --min-recent-seeds 2). Ignoră --min-success-rate')
    
    args = parser.parse_args()
    
    # Override success rate dacă --only-perfect
    if args.only_perfect:
        args.min_success_rate = 1.0
        print("🎯 Mod ONLY-PERFECT activat: doar RNG-uri cu 100%!\n")
    
    # Afișare info pentru min_recent_seeds
    if args.min_recent_seeds:
        print(f"🎯 Mod RECENT-SEEDS activat: acceptă RNG dacă găsește ultimele {args.min_recent_seeds} seeds!\n")
    
    if not args.last_n and not (args.start_year and args.end_year):
        print("❌ Specifică --last-n SAU (--start-year și --end-year)!")
        sys.exit(1)
    
    predictor = CPUOnlyPredictor(args.lottery)
    predictor.run_prediction(
        last_n=args.last_n,
        start_year=args.start_year,
        end_year=args.end_year,
        rng_timeout_minutes=args.rng_timeout,
        min_success_rate=args.min_success_rate,
        min_recent_seeds=args.min_recent_seeds
    )
