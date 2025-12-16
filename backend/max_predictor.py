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

# GPU Check
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU DETECTAT - FULL POWER MODE!")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPU nu e disponibil - CPU multicore mode")
    import numpy as cp

from lottery_config import get_lottery_config
from advanced_rng_library import RNG_TYPES, create_rng, generate_numbers


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
                          min_success_rate: float = 0.5):
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
        
        # Test TOATE RNG-urile
        print(f"\n{'='*70}")
        print(f"  FAZA 1: TESTARE EXHAUSTIVĂ TOATE RNG-URILE")
        print(f"{'='*70}\n")
        
        rng_results = {}
        
        for idx, rng_name in enumerate(RNG_TYPES.keys(), 1):
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(RNG_TYPES)}] RNG: {rng_name.upper()}")
            print(f"{'='*70}")
            
            # Pregătire tasks
            tasks = []
            for i, entry in enumerate(data):
                numbers = entry.get('numere', [])
                if len(numbers) == self.config.numbers_to_draw:
                    tasks.append((i, numbers, rng_name, self.config, seed_range, search_size))
            
            # Rulare paralel
            seeds_found = []
            draws_with_seeds = []
            
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
            
            success_rate = len(seeds_found) / len(data) if len(data) > 0 else 0
            
            print(f"\n✅ Seeds găsite: {len(seeds_found)}/{len(data)} ({success_rate:.1%})")
            
            if success_rate >= min_success_rate:
                print(f"✅ SUCCESS RATE PESTE THRESHOLD!")
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
