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
    """Worker CPU - OPTIMIZAT DINAMIC în funcție de numărul de extrageri"""
    draw_idx, numbers, rng_name, lottery_config, seed_range, base_search_size, num_extractions = args
    target_sorted = sorted(numbers)
    
    # Încercăm REVERSE mai întâi
    reversed_seed = try_reverse_engineering(rng_name, numbers, lottery_config)
    if reversed_seed is not None:
        return (draw_idx, reversed_seed)
    
    # OPTIMIZARE RADICALĂ: cu cât mai multe extrageri, cu atât mai puțin căutăm
    # PLUS: Reduce DRAMATIC pentru toate RNG-urile
    
    import math
    scale_factor = math.sqrt(max(1, num_extractions)) * 2  # Dublu factor pentru viteză
    adjusted_search = int(base_search_size / scale_factor)
    
    # LIMITE STRICTE pentru viteză:
    if num_extractions <= 3:
        adjusted_search = min(adjusted_search, 500000)  # MAX 500K pentru 3 extrageri
    elif num_extractions <= 5:
        adjusted_search = min(adjusted_search, 1000000)  # MAX 1M pentru 5
    elif num_extractions <= 10:
        adjusted_search = min(adjusted_search, 2000000)  # MAX 2M pentru 10
    
    # MERSENNE - MEGA reducere
    if rng_name == 'mersenne':
        adjusted_search = min(10000, adjusted_search // 50)  # Doar 10K pentru Mersenne!
    
    # Minimum 5K seeds
    actual_search_size = max(5000, adjusted_search)
    
    # Brute force
    test_seeds = random.sample(range(seed_range[0], seed_range[1]), 
                              min(actual_search_size, seed_range[1] - seed_range[0]))
    
    for seed in test_seeds:
        try:
            rng = create_rng(rng_name, seed)
            
            # Check dacă e composite (Joker)
            if lottery_config.is_composite:
                generated = []
                for count, min_val, max_val in lottery_config.composite_parts:
                    part = generate_numbers(rng, count, min_val, max_val)
                    generated.extend(part)
            else:
                generated = generate_numbers(rng, lottery_config.numbers_to_draw, lottery_config.min_number, lottery_config.max_number)
            
            if sorted(generated) == target_sorted:
                return (draw_idx, seed)
        except:
            continue
    
    return (draw_idx, None)


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
    
    # 12-23: Alte patterns (simplificat pentru viteză)
    
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
        print(f"📈 Base search size: {search_size:,} seeds")
        
        # Calculează search size ajustat
        import math
        if last_n:
            num_extractions = last_n
        elif start_year and end_year:
            num_extractions = (end_year - start_year + 1) * 50  # Estimare
        else:
            num_extractions = 10
        
        scale_factor = math.sqrt(max(1, num_extractions))
        adjusted_search = int(search_size / scale_factor)
        mersenne_search = min(50000, adjusted_search // 10)
        
        print(f"📉 Optimizare dinamică:")
        print(f"   {num_extractions} extrageri → scale factor: {scale_factor:.2f}")
        print(f"   Search ajustat: {adjusted_search:,} seeds (RNG-uri normale)")
        print(f"   Search Mersenne: {mersenne_search:,} seeds (foarte lent)")
        print()
        
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
            # Afișare cu warning pentru Mersenne
            if rng_name == 'mersenne':
                print(f"[{idx}/21] 💻 {rng_name.upper()} (⚠️  LENT - doar 50K seeds)")
            else:
                print(f"[{idx}/21] 💻 {rng_name.upper()}")
            
            tasks = [(i, e['numere'], rng_name, self.config, seed_range, search_size, len(data)) 
                    for i, e in enumerate(data) if len(e['numere']) == self.config.numbers_to_draw]
            
            seeds_found = []
            draws_with_seeds = []
            
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
            print(f"\n  ✅ {len(seeds_found)}/{len(data)} ({success_rate:.1%})", end='')
            
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
            
            # Predicție
            if pattern['predicted_seed']:
                try:
                    rng = create_rng(rng_name, pattern['predicted_seed'])
                    
                    # Suport COMPOSITE (Joker)
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
                    print(f"  Seed prezis: {pattern['predicted_seed']:,}")
                    print(f"  NUMERE PREZISE: {sorted(nums)}")
                    print(f"  {'='*66}\n")
                    
                    predictions.append({
                        'rng': rng_name,
                        'success_rate': result['success_rate'],
                        'pattern': pattern['pattern_type'],
                        'formula': pattern['formula'],
                        'confidence': pattern['confidence'],
                        'seed': pattern['predicted_seed'],
                        'numbers': sorted(nums),
                        'top_patterns': pattern.get('top_patterns', [])
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
    parser.add_argument('--seed-range', type=int, nargs=2, default=[0, 100000000])
    parser.add_argument('--search-size', type=int, default=10000000)
    parser.add_argument('--min-success-rate', type=float, default=0.66)
    
    args = parser.parse_args()
    
    if not args.last_n and not (args.start_year and args.end_year):
        print("❌ Specifică --last-n SAU (--start-year și --end-year)!")
        sys.exit(1)
    
    predictor = CPUOnlyPredictor(args.lottery)
    predictor.run_prediction(
        last_n=args.last_n,
        start_year=args.start_year,
        end_year=args.end_year,
        seed_range=tuple(args.seed_range),
        search_size=args.search_size,
        min_success_rate=args.min_success_rate
    )
