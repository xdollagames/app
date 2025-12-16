#!/usr/bin/env python3
"""
ULTIMATE PREDICTOR - Testează toate RNG-urile, găsește pattern-uri și face predicții
"""

import json
import sys
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
import random

from lottery_config import get_lottery_config
from advanced_rng_library import RNG_TYPES, create_rng, generate_numbers


def find_seed_for_draw_worker(args):
    """Worker pentru găsirea seed-ului pentru o extragere cu un RNG specific"""
    draw_idx, numbers, rng_type, lottery_config, seed_range, search_size = args
    target_sorted = sorted(numbers)
    
    # Generează seed-uri random
    test_seeds = random.sample(range(seed_range[0], seed_range[1]), 
                              min(search_size, seed_range[1] - seed_range[0]))
    
    for seed in test_seeds:
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
    """Analizează pattern-ul matematic al seed-urilor - TOATE TIPURILE"""
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
    
    # 1. Pattern LINEAR
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
    
    # 2. Pattern POLINOMIAL grad 2
    try:
        poly_coeffs = np.polyfit(x, y, 2)
        poly_pred = np.poly1d(poly_coeffs)(len(seeds))
        poly_error = np.mean(np.abs(y - np.poly1d(poly_coeffs)(x)))
        all_patterns['polynomial_2'] = {
            'pred': poly_pred,
            'error': poly_error,
            'formula': f"y = {poly_coeffs[0]:.2e}*x² + {poly_coeffs[1]:.2f}*x + {poly_coeffs[2]:.2f}"
        }
    except:
        all_patterns['polynomial_2'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    
    # 3. Pattern LCG CHAIN: seed[n] = (a*seed[n-1] + c) % m
    if len(seeds) >= 2:
        try:
            m_estimate = max(seeds) * 2 if max(seeds) > 0 else 2147483648
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
            
            all_patterns['lcg_chain'] = {
                'pred': lcg_pred,
                'error': lcg_error,
                'formula': f"S(n+1) = ({a:.4f}*S(n) + {c:.2f}) mod {int(m_estimate)}"
            }
        except:
            all_patterns['lcg_chain'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['lcg_chain'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 4. Pattern MODULAR ARITHMETIC
    if len(seeds) >= 2:
        try:
            diffs = np.diff(seeds)
            avg_diff = np.mean(diffs)
            m_estimate = max(seeds) * 2 if max(seeds) > 0 else 2147483648
            
            modular_pred = (seeds[-1] + avg_diff) % m_estimate
            
            errors = []
            for i in range(1, len(seeds)):
                pred_val = (seeds[i-1] + avg_diff) % m_estimate
                errors.append(abs(pred_val - seeds[i]))
            modular_error = np.mean(errors)
            
            all_patterns['modular'] = {
                'pred': modular_pred,
                'error': modular_error,
                'formula': f"S(n+1) = (S(n) + {avg_diff:.2f}) mod {int(m_estimate)}"
            }
        except:
            all_patterns['modular'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['modular'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 5. Pattern FIBONACCI-like
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
                'formula': f"S(n) = {a:.4f}*S(n-1) + {b:.4f}*S(n-2)"
            }
        except:
            all_patterns['fibonacci'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
    else:
        all_patterns['fibonacci'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
    # 6. Pattern CONSTANT DIFFERENCE
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
    else:
        all_patterns['const_diff'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
    
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
    
    # Calculăm confidence
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
            'pred': int(round(v['pred'])) if v['pred'] is not None else None
        } for k, v in all_patterns.items()}
    }


class UltimatePredictor:
    def __init__(self, lottery_type: str = "5-40"):
        self.lottery_type = lottery_type
        self.config = get_lottery_config(lottery_type)
        self.data_file = f"{lottery_type}_data.json"
        
    def load_data(self, last_n: Optional[int] = None, 
                  start_year: Optional[int] = None, 
                  end_year: Optional[int] = None) -> List[Dict]:
        """Încarcă datele - fie ultimele N, fie interval de ani"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Fișierul {self.data_file} nu există!")
            print(f"\n💡 Trebuie să scrape-zi datele mai întâi:")
            print(f"   python3 unified_lottery_scraper.py --lottery {self.lottery_type} --year all\n")
            sys.exit(1)
        
        # Detectează formatul
        if isinstance(data, dict) and 'draws' in data:
            all_data = data['draws']
        elif isinstance(data, list):
            all_data = data
        else:
            print(f"❌ Format de date necunoscut în {self.data_file}")
            sys.exit(1)
        
        # Filtrare
        if last_n is not None:
            # Ultimele N extrageri
            filtered_data = all_data[-last_n:] if len(all_data) >= last_n else all_data
        elif start_year is not None and end_year is not None:
            # Interval de ani
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
                except (ValueError, IndexError):
                    continue
        else:
            filtered_data = all_data
        
        # Normalizare format
        normalized = []
        for entry in filtered_data:
            normalized.append({
                'data': entry.get('data', entry.get('date', 'N/A')),
                'numere': entry.get('numere', entry.get('numbers', entry.get('numbers_sorted', [])))
            })
        
        return normalized
    
    def test_all_rngs(self, data: List[Dict], seed_range: tuple = (0, 10000000), 
                     search_size: int = 2000000, min_success_rate: float = 0.5):
        """Testează toate RNG-urile și returnează cele mai bune"""
        num_cores = cpu_count()
        
        print(f"💻 CPU cores: {num_cores}")
        print(f"🔍 Seed range: {seed_range[0]:,} - {seed_range[1]:,}")
        print(f"📊 Search size: {search_size:,} seeds/extragere")
        print(f"🎯 Min success rate: {min_success_rate:.1%}\n")
        
        rng_results = {}
        
        for rng_name in RNG_TYPES.keys():
            print(f"{'='*70}")
            print(f"🔍 Testare RNG: {rng_name.upper()}")
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
                for i, result in enumerate(pool.imap_unordered(find_seed_for_draw_worker, tasks, chunksize=optimal_chunksize)):
                    idx, seed = result
                    
                    if seed is not None:
                        seeds_found.append(seed)
                        draws_with_seeds.append({
                            'idx': idx,
                            'date': data[idx]['data'],
                            'numbers': data[idx]['numere'],
                            'seed': seed
                        })
                    
                    # Progress
                    if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                        progress = 100 * (i + 1) / len(tasks)
                        print(f"  Procesate {i + 1}/{len(tasks)} ({progress:.1f}%)... {len(seeds_found)} seeds găsite", end='\r')
            
            success_rate = len(seeds_found) / len(data) if len(data) > 0 else 0
            
            print(f"\n✅ Seed-uri găsite: {len(seeds_found)}/{len(data)} ({success_rate:.1%})")
            
            if success_rate >= min_success_rate:
                print(f"✅ SUCCESS RATE peste threshold!")
                rng_results[rng_name] = {
                    'seeds': seeds_found,
                    'draws': draws_with_seeds,
                    'success_rate': success_rate
                }
            else:
                print(f"❌ Success rate sub threshold ({success_rate:.1%} < {min_success_rate:.1%})")
            
            print()
        
        return rng_results
    
    def run_ultimate_prediction(self, last_n: Optional[int] = None,
                                start_year: Optional[int] = None,
                                end_year: Optional[int] = None,
                                seed_range: tuple = (0, 10000000),
                                search_size: int = 2000000,
                                min_success_rate: float = 0.5):
        """Rulează predicția ultimate"""
        
        print(f"\n{'='*70}")
        print(f"  🎯 ULTIMATE PREDICTOR - {self.lottery_type.upper()}")
        print(f"{'='*70}\n")
        
        # Încărcare date
        if last_n:
            print(f"📊 Încărcare ultimele {last_n} extrageri...")
            data = self.load_data(last_n=last_n)
        else:
            print(f"📊 Încărcare date pentru {start_year}-{end_year}...")
            data = self.load_data(start_year=start_year, end_year=end_year)
        
        print(f"✅ {len(data)} extrageri încărcate\n")
        
        # Afișează extragerile
        print(f"📋 Extrageri încărcate:")
        for i, entry in enumerate(data, 1):
            print(f"  {i}. {entry['data']:15s} → {entry['numere']}")
        print()
        
        # Testare toate RNG-urile
        print(f"{'='*70}")
        print(f"  FAZA 1: TESTARE TOATE RNG-URILE ({len(RNG_TYPES)} algoritmi)")
        print(f"{'='*70}\n")
        
        rng_results = self.test_all_rngs(data, seed_range, search_size, min_success_rate)
        
        if not rng_results:
            print(f"\n❌ Niciun RNG nu a trecut de threshold-ul de {min_success_rate:.1%}!")
            return
        
        # Analiză pattern-uri pentru fiecare RNG bun
        print(f"\n{'='*70}")
        print(f"  FAZA 2: ANALIZĂ PATTERN-URI ȘI PREDICȚII")
        print(f"{'='*70}\n")
        
        predictions = []
        
        for rng_name, result in sorted(rng_results.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            print(f"\n{'='*70}")
            print(f"  RNG: {rng_name.upper()}")
            print(f"  Success rate: {result['success_rate']:.1%}")
            print(f"{'='*70}\n")
            
            # Analiză pattern
            pattern_analysis = analyze_seed_pattern(result['seeds'])
            
            print(f"📊 Pattern detectat: {pattern_analysis['pattern_type'].upper()}")
            print(f"📐 Formula: {pattern_analysis['formula']}")
            print(f"🎯 Confidence: {pattern_analysis['confidence']:.2f}%")
            print(f"❌ Error: {pattern_analysis.get('error', 'N/A')}\n")
            
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
                    print(f"  🎯 PREDICȚIE")
                    print(f"{'='*70}")
                    print(f"  Seed prezis: {pattern_analysis['predicted_seed']:,}")
                    print(f"  NUMERE PREZISE: {sorted(predicted_numbers)}")
                    print(f"{'='*70}\n")
                    
                    predictions.append({
                        'rng': rng_name,
                        'success_rate': result['success_rate'],
                        'pattern': pattern_analysis['pattern_type'],
                        'formula': pattern_analysis['formula'],
                        'confidence': pattern_analysis['confidence'],
                        'seed': pattern_analysis['predicted_seed'],
                        'numbers': sorted(predicted_numbers)
                    })
                except Exception as e:
                    print(f"❌ Eroare la generare predicție: {e}\n")
            else:
                print(f"❌ Nu s-a putut genera predicție\n")
        
        # Sumar final
        if predictions:
            print(f"\n{'='*70}")
            print(f"  📊 SUMAR PREDICȚII")
            print(f"{'='*70}\n")
            
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. RNG: {pred['rng']}")
                print(f"   Success: {pred['success_rate']:.1%} | Confidence: {pred['confidence']:.1f}%")
                print(f"   Pattern: {pred['pattern']}")
                print(f"   Numere: {pred['numbers']}\n")
            
            # Salvare rezultate
            output_file = f"ultimate_prediction_{self.lottery_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'lottery': self.lottery_type,
                    'timestamp': datetime.now().isoformat(),
                    'data_size': len(data),
                    'predictions': predictions
                }, f, indent=2)
            
            print(f"💾 Rezultate salvate: {output_file}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultimate Predictor - testează toate RNG-urile și face predicții')
    parser.add_argument('--lottery', default='5-40', choices=['5-40', '6-49', 'joker'],
                      help='Tip loterie (default: 5-40)')
    parser.add_argument('--last-n', type=int,
                      help='Ultimele N extrageri (ex: --last-n 10)')
    parser.add_argument('--start-year', type=int,
                      help='An început (folosit doar dacă nu e --last-n)')
    parser.add_argument('--end-year', type=int,
                      help='An sfârșit (folosit doar dacă nu e --last-n)')
    parser.add_argument('--seed-range', type=int, nargs=2, default=[0, 10000000],
                      help='Range seed-uri (default: 0 10000000)')
    parser.add_argument('--search-size', type=int, default=2000000,
                      help='Seeds testate per extragere (default: 2000000)')
    parser.add_argument('--min-success-rate', type=float, default=0.5,
                      help='Min success rate pentru un RNG (default: 0.5)')
    
    args = parser.parse_args()
    
    if not args.last_n and not (args.start_year and args.end_year):
        print("❌ Trebuie să specifici fie --last-n, fie --start-year și --end-year!")
        sys.exit(1)
    
    predictor = UltimatePredictor(args.lottery)
    predictor.run_ultimate_prediction(
        last_n=args.last_n,
        start_year=args.start_year,
        end_year=args.end_year,
        seed_range=tuple(args.seed_range),
        search_size=args.search_size,
        min_success_rate=args.min_success_rate
    )
