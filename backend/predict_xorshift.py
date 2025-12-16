#!/usr/bin/env python3
"""
Script pentru investigarea xorshift_simple pe perioada 2010-2025
și prezicerea următoarei secvențe de numere.
OPTIMIZAT PENTRU MULTICORE CPU
"""

import json
import sys
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count

from lottery_config import get_lottery_config
from advanced_rng_library import create_rng, generate_numbers


def find_seed_worker(args):
    """Worker function pentru multiprocessing - căutare optimizată"""
    idx, numbers, lottery_config, seed_range, search_size = args
    target_sorted = sorted(numbers)
    
    # Generează seed-uri random în loc să caute secvențial (mult mai eficient)
    import random
    test_seeds = random.sample(range(seed_range[0], seed_range[1]), 
                              min(search_size, seed_range[1] - seed_range[0]))
    
    for seed in test_seeds:
        try:
            rng = create_rng('xorshift_simple', seed)
            generated = generate_numbers(
                rng,
                lottery_config.numbers_to_draw,
                lottery_config.min_number,
                lottery_config.max_number
            )
            if sorted(generated) == target_sorted:
                return (idx, seed)
        except:
            continue
    
    return (idx, None)


class XorshiftInvestigator:
    def __init__(self, lottery_type: str = "5-40"):
        self.lottery_type = lottery_type
        self.config = get_lottery_config(lottery_type)
        self.data_file = f"{lottery_type}_data.json"
        
    def load_data(self, start_year: int = 2010, end_year: int = 2025) -> List[Dict]:
        """Încarcă datele și filtrează după perioadă"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Fișierul {self.data_file} nu există!")
            print(f"\n💡 Trebuie să scrape-zi datele mai întâi:")
            print(f"   python3 unified_lottery_scraper.py --lottery {self.lottery_type} --year all\n")
            sys.exit(1)
        
        # Detectează formatul datelor
        if isinstance(data, dict) and 'draws' in data:
            # Format nou cu wrapper
            all_data = data['draws']
        elif isinstance(data, list):
            # Format vechi, listă directă
            all_data = data
        else:
            print(f"❌ Format de date necunoscut în {self.data_file}")
            sys.exit(1)
            
        filtered_data = []
        for entry in all_data:
            try:
                # Suport pentru ambele formate de dată
                date_str = entry.get('data', entry.get('date', ''))
                
                # Extrage anul din diverse formate
                if '.' in date_str:
                    year = int(date_str.split('.')[-1])
                elif '-' in date_str:
                    year = int(date_str.split('-')[0])
                else:
                    year = entry.get('year', 0)
                    
                if start_year <= year <= end_year:
                    # Normalizează formatul
                    normalized = {
                        'data': date_str,
                        'numere': entry.get('numere', entry.get('numbers', entry.get('numbers_sorted', [])))
                    }
                    filtered_data.append(normalized)
            except (ValueError, IndexError):
                continue
                
        return filtered_data
    
    def analyze_seed_pattern(self, seeds: List[int]) -> Dict:
        """Analizează pattern-ul matematic al seed-urilor - TOATE TIPURILE"""
        if len(seeds) < 3:
            return {
                'pattern_type': 'insufficient_data',
                'predicted_seed': None,
                'confidence': 0
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
                'formula': f"y = {linear_coeffs[0]:.2f}*x + {linear_coeffs[1]:.2f}"
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
                'formula': f"y = {poly_coeffs[0]:.2f}*x² + {poly_coeffs[1]:.2f}*x + {poly_coeffs[2]:.2f}"
            }
        except:
            all_patterns['polynomial_2'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        
        # 3. Pattern POLINOMIAL grad 3: y = a*x³ + b*x² + c*x + d
        try:
            poly3_coeffs = np.polyfit(x, y, 3)
            poly3_pred = np.poly1d(poly3_coeffs)(len(seeds))
            poly3_error = np.mean(np.abs(y - np.poly1d(poly3_coeffs)(x)))
            all_patterns['polynomial_3'] = {
                'pred': poly3_pred,
                'error': poly3_error,
                'formula': f"y = {poly3_coeffs[0]:.2f}*x³ + ... (grad 3)"
            }
        except:
            all_patterns['polynomial_3'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        
        # 4. Pattern EXPONENȚIAL: y = a*e^(b*x) + c
        try:
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c
            
            popt, _ = curve_fit(exp_func, x, y, maxfev=5000)
            exp_pred = exp_func(len(seeds), *popt)
            exp_error = np.mean(np.abs(y - exp_func(x, *popt)))
            all_patterns['exponential'] = {
                'pred': exp_pred,
                'error': exp_error,
                'formula': f"y = {popt[0]:.2f}*e^({popt[1]:.4f}*x) + {popt[2]:.2f}"
            }
        except:
            all_patterns['exponential'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        
        # 5. Pattern FIBONACCI-like: seed[n] = a*seed[n-1] + b*seed[n-2]
        if len(seeds) >= 3:
            try:
                # Construim sistem de ecuații pentru ultimi 3 seeds
                # seed[n] = a*seed[n-1] + b*seed[n-2]
                A = np.array([[seeds[i-1], seeds[i-2]] for i in range(2, len(seeds))])
                B = np.array([seeds[i] for i in range(2, len(seeds))])
                coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
                a, b = coeffs
                
                # Predicție
                fib_pred = a * seeds[-1] + b * seeds[-2]
                
                # Eroare pe toți seeds
                errors = []
                for i in range(2, len(seeds)):
                    pred_val = a * seeds[i-1] + b * seeds[i-2]
                    errors.append(abs(pred_val - seeds[i]))
                fib_error = np.mean(errors) if errors else float('inf')
                
                all_patterns['fibonacci'] = {
                    'pred': fib_pred,
                    'error': fib_error,
                    'formula': f"seed[n] = {a:.4f}*seed[n-1] + {b:.4f}*seed[n-2]"
                }
            except:
                all_patterns['fibonacci'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        else:
            all_patterns['fibonacci'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
        
        # 6. Pattern LCG (Linear Congruential): seed[n] = (a*seed[n-1] + c) % m
        if len(seeds) >= 2:
            try:
                # Estimăm m ca fiind puțin mai mare decât max seed
                m_estimate = max(seeds) * 2
                
                # seed[n] ≈ a*seed[n-1] + c (ignorăm modulo pentru estimare)
                X = np.array([[seeds[i-1], 1] for i in range(1, len(seeds))])
                Y = np.array([seeds[i] for i in range(1, len(seeds))])
                coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                a, c = coeffs
                
                # Predicție
                lcg_pred = (a * seeds[-1] + c) % m_estimate
                
                # Eroare
                errors = []
                for i in range(1, len(seeds)):
                    pred_val = (a * seeds[i-1] + c) % m_estimate
                    errors.append(abs(pred_val - seeds[i]))
                lcg_error = np.mean(errors) if errors else float('inf')
                
                all_patterns['lcg_chain'] = {
                    'pred': lcg_pred,
                    'error': lcg_error,
                    'formula': f"seed[n] = ({a:.4f}*seed[n-1] + {c:.2f}) mod {m_estimate}"
                }
            except:
                all_patterns['lcg_chain'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        else:
            all_patterns['lcg_chain'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
        
        # 7. Pattern MODULAR ARITHMETIC: seed[n] = (seed[n-1] + delta) % m
        if len(seeds) >= 2:
            try:
                # Calculăm diferențele
                diffs = np.diff(seeds)
                avg_diff = np.mean(diffs)
                m_estimate = max(seeds) * 2
                
                modular_pred = (seeds[-1] + avg_diff) % m_estimate
                
                # Eroare
                errors = []
                for i in range(1, len(seeds)):
                    pred_val = (seeds[i-1] + avg_diff) % m_estimate
                    errors.append(abs(pred_val - seeds[i]))
                modular_error = np.mean(errors)
                
                all_patterns['modular_arithmetic'] = {
                    'pred': modular_pred,
                    'error': modular_error,
                    'formula': f"seed[n] = (seed[n-1] + {avg_diff:.2f}) mod {m_estimate}"
                }
            except:
                all_patterns['modular_arithmetic'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        else:
            all_patterns['modular_arithmetic'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
        
        # 8. Pattern DIFERENȚE CONSTANTE (arithmetic progression)
        if len(seeds) >= 2:
            try:
                diffs = np.diff(seeds)
                avg_diff = np.mean(diffs)
                const_diff_pred = seeds[-1] + avg_diff
                const_diff_error = np.std(diffs)
                
                all_patterns['constant_difference'] = {
                    'pred': const_diff_pred,
                    'error': const_diff_error,
                    'formula': f"seed[n] = seed[n-1] + {avg_diff:.2f}"
                }
            except:
                all_patterns['constant_difference'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        else:
            all_patterns['constant_difference'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
        
        # 9. Pattern RATIE CONSTANTĂ (geometric progression)
        if len(seeds) >= 2 and all(s > 0 for s in seeds):
            try:
                ratios = [seeds[i] / seeds[i-1] for i in range(1, len(seeds))]
                avg_ratio = np.mean(ratios)
                ratio_pred = seeds[-1] * avg_ratio
                ratio_error = np.std(ratios) * seeds[-1]
                
                all_patterns['constant_ratio'] = {
                    'pred': ratio_pred,
                    'error': ratio_error,
                    'formula': f"seed[n] = seed[n-1] * {avg_ratio:.4f}"
                }
            except:
                all_patterns['constant_ratio'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        else:
            all_patterns['constant_ratio'] = {'pred': None, 'error': float('inf'), 'formula': 'insufficient_data'}
        
        # 10. Pattern LOGARITMIC: y = a*log(x) + b
        try:
            log_x = np.log(x + 1)  # +1 pentru a evita log(0)
            log_coeffs = np.polyfit(log_x, y, 1)
            log_pred_x = np.log(len(seeds) + 1)
            log_pred = log_coeffs[0] * log_pred_x + log_coeffs[1]
            log_error = np.mean(np.abs(y - (log_coeffs[0] * log_x + log_coeffs[1])))
            
            all_patterns['logarithmic'] = {
                'pred': log_pred,
                'error': log_error,
                'formula': f"y = {log_coeffs[0]:.2f}*log(x) + {log_coeffs[1]:.2f}"
            }
        except:
            all_patterns['logarithmic'] = {'pred': None, 'error': float('inf'), 'formula': 'failed'}
        
        # Selectare cel mai bun pattern (cel cu cea mai mică eroare)
        valid_patterns = {k: v for k, v in all_patterns.items() 
                         if v['pred'] is not None and not np.isnan(v['error'])}
        
        if not valid_patterns:
            return {
                'pattern_type': 'no_valid_pattern',
                'predicted_seed': None,
                'confidence': 0,
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
                'error': round(v['error'], 2) if not np.isnan(v['error']) else 'inf',
                'formula': v['formula'],
                'pred': int(round(v['pred'])) if v['pred'] is not None else None
            } for k, v in all_patterns.items()}
        }
    
    def run_investigation(self, start_year: int = 2010, end_year: int = 2025, 
                         seed_range: tuple = (0, 10000000), search_size: int = 2000000):
        """Rulează investigația completă"""
        num_cores = cpu_count()
        
        print(f"\n{'='*70}")
        print(f"  INVESTIGAȚIE XORSHIFT_SIMPLE - {self.lottery_type.upper()}")
        print(f"{'='*70}\n")
        
        print(f"💻 CPU cores utilizate: {num_cores}")
        print(f"🔍 Seed range: {seed_range[0]:,} - {seed_range[1]:,}")
        print(f"📊 Search size: {search_size:,} seeds testate per extragere")
        print(f"📅 Încărcare date pentru perioada {start_year}-{end_year}...")
        data = self.load_data(start_year, end_year)
        print(f"✅ {len(data)} extrageri încărcate\n")
        
        print("🔍 Căutare seed-uri PARALLEL pe toate cores-urile...")
        print("(Aceasta poate dura câteva minute...)\n")
        
        # Pregătește task-urile pentru multiprocessing
        tasks = []
        for i, entry in enumerate(data):
            numbers = entry.get('numere', [])
            if len(numbers) == self.config.numbers_to_draw:
                tasks.append((i, numbers, self.config, seed_range, search_size))
        
        # Rulează parallel
        seeds_found = []
        draws_with_seeds = []
        
        # Optimizare chunksize pentru numărul de core-uri
        optimal_chunksize = max(1, len(tasks) // (num_cores * 4))
        
        with Pool(processes=num_cores) as pool:
            results = []
            for i, result in enumerate(pool.imap_unordered(find_seed_worker, tasks, chunksize=optimal_chunksize)):
                idx, seed = result
                
                if seed is not None:
                    seeds_found.append(seed)
                    draws_with_seeds.append({
                        'date': data[idx]['data'],
                        'numbers': data[idx]['numere'],
                        'seed': seed
                    })
                
                # Progress update (mai frecvent pentru mai multe core-uri)
                update_freq = max(10, len(tasks) // 20)
                if (i + 1) % update_freq == 0 or (i + 1) == len(tasks):
                    progress = 100 * (i + 1) / len(tasks)
                    print(f"  Procesate {i + 1}/{len(tasks)} extrageri ({progress:.1f}%)... {len(seeds_found)} seed-uri găsite")
        
        print(f"\n✅ Procesare completă!")
        print(f"📈 Seed-uri găsite: {len(seeds_found)}/{len(data)} ({100*len(seeds_found)/len(data):.1f}%)\n")
        
        if len(seeds_found) < 3:
            print("❌ Prea puține seed-uri găsite pentru analiza pattern-ului!")
            return
        
        # Analiză pattern
        print("🧮 Analiză pattern matematic al seed-urilor...")
        pattern_analysis = self.analyze_seed_pattern(seeds_found)
        
        # Generare numere prezise
        predicted_seed = pattern_analysis['predicted_seed']
        if predicted_seed:
            rng = create_rng('xorshift_simple', predicted_seed)
            predicted_numbers = generate_numbers(
                rng,
                self.config.numbers_to_draw,
                self.config.min_number,
                self.config.max_number
            )
        else:
            predicted_numbers = []
        
        # Output rezultate
        self._print_results(
            start_year, end_year, len(data), 
            draws_with_seeds, seeds_found, 
            pattern_analysis, predicted_numbers
        )
        
        # Salvare JSON
        self._save_json_results(
            start_year, end_year, len(data),
            draws_with_seeds, seeds_found,
            pattern_analysis, predicted_numbers
        )
    
    def _print_results(self, start_year, end_year, total_draws, 
                      draws_with_seeds, seeds_found, pattern_analysis, predicted_numbers):
        """Afișează rezultatele în terminal"""
        print(f"\n{'='*70}")
        print(f"  REZULTATE INVESTIGAȚIE")
        print(f"{'='*70}\n")
        
        print(f"📅 Perioadă analizată: {start_year}-{end_year}")
        print(f"🎰 Loterie: {self.lottery_type.upper()}")
        print(f"🔢 Total extrageri: {total_draws}")
        print(f"✅ Seed-uri găsite: {len(seeds_found)} ({100*len(seeds_found)/total_draws:.1f}%)\n")
        
        if seeds_found:
            print(f"📊 Statistici seed-uri:")
            print(f"   Min: {min(seeds_found)}")
            print(f"   Max: {max(seeds_found)}")
            print(f"   Medie: {np.mean(seeds_found):.2f}\n")
        
        print(f"🧮 Pattern detectat: {pattern_analysis['pattern_type'].upper()}")
        print(f"📈 Grad de încredere: {pattern_analysis['confidence']:.2f}%\n")
        
        if predicted_numbers:
            print(f"{'='*70}")
            print(f"  🎯 PREDICȚIE URMĂTOAREA EXTRAGERE")
            print(f"{'='*70}\n")
            print(f"   Seed prezis: {pattern_analysis['predicted_seed']}")
            print(f"   NUMERE PREZISE: {sorted(predicted_numbers)}\n")
        
        print(f"{'='*70}\n")
    
    def _save_json_results(self, start_year, end_year, total_draws,
                          draws_with_seeds, seeds_found, pattern_analysis, predicted_numbers):
        """Salvează rezultatele în JSON"""
        output_file = f"xorshift_investigation_{start_year}_{end_year}.json"
        
        results = {
            'lottery_type': self.lottery_type,
            'rng': 'xorshift_simple',
            'analysis_timestamp': datetime.now().isoformat(),
            'period': {
                'start_year': start_year,
                'end_year': end_year,
                'total_draws': total_draws
            },
            'seeds_found': {
                'count': len(seeds_found),
                'percentage': round(100 * len(seeds_found) / total_draws, 2) if total_draws > 0 else 0,
                'values': seeds_found,
                'draws': draws_with_seeds
            },
            'pattern_analysis': pattern_analysis,
            'prediction': {
                'seed': pattern_analysis['predicted_seed'],
                'numbers': sorted(predicted_numbers) if predicted_numbers else []
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Rezultate salvate în: {output_file}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Investigare xorshift_simple și predicție')
    parser.add_argument('--lottery', default='5-40', choices=['5-40', '6-49', 'joker'],
                      help='Tip loterie (default: 5-40)')
    parser.add_argument('--start-year', type=int, default=2010,
                      help='An început (default: 2010)')
    parser.add_argument('--end-year', type=int, default=2025,
                      help='An sfârșit (default: 2025)')
    parser.add_argument('--seed-range', type=int, nargs=2, default=[0, 10000000],
                      help='Range seed-uri (default: 0 10000000)')
    parser.add_argument('--search-size', type=int, default=2000000,
                      help='Câte seeds să testeze per extragere (default: 2000000)')
    
    args = parser.parse_args()
    
    investigator = XorshiftInvestigator(args.lottery)
    investigator.run_investigation(
        args.start_year, 
        args.end_year,
        tuple(args.seed_range),
        args.search_size
    )
