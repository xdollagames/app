#!/usr/bin/env python3
"""
Script pentru investigarea xorshift_simple pe perioada 2010-2025
și prezicerea următoarei secvențe de numere.
"""

import json
import sys
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.optimize import curve_fit

from lottery_config import LOTTERY_CONFIGS
from advanced_rng_library import RNGLibrary


class XorshiftInvestigator:
    def __init__(self, lottery_type: str = "5-40"):
        self.lottery_type = lottery_type
        self.config = LOTTERY_CONFIGS[lottery_type]
        self.rng_lib = RNGLibrary()
        self.data_file = f"{lottery_type}_data.json"
        
    def load_data(self, start_year: int = 2010, end_year: int = 2025) -> List[Dict]:
        """Încarcă datele și filtrează după perioadă"""
        try:
            with open(self.data_file, 'r') as f:
                all_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Fișierul {self.data_file} nu există!")
            sys.exit(1)
            
        filtered_data = []
        for entry in all_data:
            try:
                date_str = entry.get('data', '')
                year = int(date_str.split('.')[-1])
                if start_year <= year <= end_year:
                    filtered_data.append(entry)
            except (ValueError, IndexError):
                continue
                
        return filtered_data
    
    def find_seed_for_draw(self, numbers: List[int], max_seed: int = 1000000) -> Optional[int]:
        """Găsește seed-ul care generează numerele date folosind xorshift_simple"""
        target_sorted = sorted(numbers)
        
        for seed in range(1, max_seed):
            try:
                generated = self.rng_lib.xorshift_simple(
                    seed, 
                    self.config['numbers_to_draw'], 
                    self.config['max_number']
                )
                if sorted(generated) == target_sorted:
                    return seed
            except:
                continue
                
        return None
    
    def analyze_seed_pattern(self, seeds: List[int]) -> Dict:
        """Analizează pattern-ul matematic al seed-urilor"""
        if len(seeds) < 3:
            return {
                'pattern_type': 'insufficient_data',
                'predicted_seed': None,
                'confidence': 0
            }
        
        x = np.arange(len(seeds))
        y = np.array(seeds)
        
        # Testare pattern linear
        try:
            linear_coeffs = np.polyfit(x, y, 1)
            linear_pred = np.poly1d(linear_coeffs)(len(seeds))
            linear_error = np.mean(np.abs(y - np.poly1d(linear_coeffs)(x)))
        except:
            linear_pred = None
            linear_error = float('inf')
        
        # Testare pattern polinomial (grad 2)
        try:
            poly_coeffs = np.polyfit(x, y, 2)
            poly_pred = np.poly1d(poly_coeffs)(len(seeds))
            poly_error = np.mean(np.abs(y - np.poly1d(poly_coeffs)(x)))
        except:
            poly_pred = None
            poly_error = float('inf')
        
        # Testare pattern exponential
        try:
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c
            
            popt, _ = curve_fit(exp_func, x, y, maxfev=5000)
            exp_pred = exp_func(len(seeds), *popt)
            exp_error = np.mean(np.abs(y - exp_func(x, *popt)))
        except:
            exp_pred = None
            exp_error = float('inf')
        
        # Selectare cel mai bun pattern
        errors = {
            'linear': linear_error,
            'polynomial': poly_error,
            'exponential': exp_error
        }
        
        best_pattern = min(errors, key=errors.get)
        
        if best_pattern == 'linear':
            predicted_seed = int(round(linear_pred))
            confidence = max(0, min(100, 100 * (1 - linear_error / np.mean(y))))
        elif best_pattern == 'polynomial':
            predicted_seed = int(round(poly_pred))
            confidence = max(0, min(100, 100 * (1 - poly_error / np.mean(y))))
        else:
            predicted_seed = int(round(exp_pred))
            confidence = max(0, min(100, 100 * (1 - exp_error / np.mean(y))))
        
        return {
            'pattern_type': best_pattern,
            'predicted_seed': predicted_seed,
            'confidence': round(confidence, 2),
            'all_errors': {k: round(v, 2) for k, v in errors.items()}
        }
    
    def run_investigation(self, start_year: int = 2010, end_year: int = 2025):
        """Rulează investigația completă"""
        print(f"\n{'='*70}")
        print(f"  INVESTIGAȚIE XORSHIFT_SIMPLE - {self.lottery_type.upper()}")
        print(f"{'='*70}\n")
        
        print(f"📊 Încărcare date pentru perioada {start_year}-{end_year}...")
        data = self.load_data(start_year, end_year)
        print(f"✅ {len(data)} extrageri încărcate\n")
        
        print("🔍 Căutare seed-uri pentru fiecare extragere...")
        print("(Aceasta poate dura câteva minute...)\n")
        
        seeds_found = []
        draws_with_seeds = []
        
        for i, entry in enumerate(data):
            numbers = entry.get('numere', [])
            date_str = entry.get('data', 'N/A')
            
            if len(numbers) != self.config['numbers_to_draw']:
                continue
            
            seed = self.find_seed_for_draw(numbers)
            
            if seed is not None:
                seeds_found.append(seed)
                draws_with_seeds.append({
                    'date': date_str,
                    'numbers': numbers,
                    'seed': seed
                })
                
            if (i + 1) % 50 == 0:
                print(f"  Procesate {i + 1}/{len(data)} extrageri... ({len(seeds_found)} seed-uri găsite)")
        
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
            predicted_numbers = self.rng_lib.xorshift_simple(
                predicted_seed,
                self.config['numbers_to_draw'],
                self.config['max_number']
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
    
    args = parser.parse_args()
    
    investigator = XorshiftInvestigator(args.lottery)
    investigator.run_investigation(args.start_year, args.end_year)
