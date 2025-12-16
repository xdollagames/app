#!/usr/bin/env python3
"""
Unified Pragmatic Pattern Finder - pentru TOATE loteriile

Utilizare:
    # Test pe Loto 6/49
    python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_data.json --min-matches 3

    # Test pe Joker
    python3 unified_pattern_finder.py --lottery joker --input joker_data.json --min-matches 3

    # Quick test
    python3 unified_pattern_finder.py --lottery 5-40 --input loto_data.json --quick-test
"""

import argparse
import json
import time
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count
import numpy as np
from collections import Counter
import sys

from advanced_rng_library import RNG_TYPES, create_rng, generate_numbers
from advanced_pattern_finder import AdvancedPatternFinder
from lottery_config import get_lottery_config, list_available_lotteries


def evaluate_seed_for_composite(seed: int, target: List[int], rng_type: str, 
                                composite_parts: List[Tuple[int, int, int]]) -> Dict:
    """Evaluează seed pentru loterii compuse (ex: Joker)"""
    try:
        rng = create_rng(rng_type, seed)
        
        # Generează pentru fiecare parte
        generated = []
        total_matches = 0
        
        for count, min_val, max_val in composite_parts:
            part_nums = generate_numbers(rng, count, min_val, max_val)
            generated.extend(part_nums)
        
        # Calculează matches
        matches = len(set(generated) & set(target))
        
        return {
            'seed': seed,
            'matches': matches,
            'generated': generated,
            'score': matches / len(target)
        }
    except:
        return None


def find_best_seed_for_draw(args):
    """Pentru o extragere, găsește BEST seed pentru un RNG"""
    draw_idx, target, rng_type, config, seed_range, search_size, min_matches = args
    
    import random
    test_seeds = random.sample(range(seed_range[0], seed_range[1]),
                               min(search_size, seed_range[1] - seed_range[0]))
    
    best_result = None
    best_matches = 0
    
    target_set = set(target)
    
    for seed in test_seeds:
        try:
            rng = create_rng(rng_type, seed)
            
            # Generează numere - dacă e compus, folosește composite_parts
            if config.is_composite:
                generated = []
                for count, min_val, max_val in config.composite_parts:
                    part_nums = generate_numbers(rng, count, min_val, max_val)
                    generated.extend(part_nums)
            else:
                generated = generate_numbers(rng, config.numbers_to_draw, 
                                            config.min_number, config.max_number)
            
            matches = len(set(generated) & target_set)
            
            if matches >= min_matches and matches > best_matches:
                best_matches = matches
                best_result = {
                    'seed': seed,
                    'matches': matches,
                    'generated': generated,
                    'score': matches / len(target)
                }
                
                if matches == len(target):  # Perfect!
                    break
        except:
            continue
    
    return draw_idx, best_result


class UnifiedPatternFinder:
    def __init__(self, data_file: str, lottery_type: str):
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        self.config = get_lottery_config(lottery_type)
        self.draws = data['draws']
        
        print(f"Încărcat {len(self.draws)} extrageri pentru {self.config.name}\n")
    
    def find_pragmatic_patterns(self,
                                rng_types: List[str] = None,
                                seed_range: tuple = (0, 10000000),
                                search_size: int = 2000000,
                                min_matches: int = 3,
                                success_threshold: float = 0.65,
                                workers: int = None):
        """
        Găsește patterns "pragmatice" - good enough, nu perfect
        """
        if rng_types is None:
            rng_types = list(RNG_TYPES.keys())
        
        if workers is None:
            workers = cpu_count()
        
        print(f"{'='*70}")
        print(f"PRAGMATIC PATTERN FINDER - {self.config.name}")
        print(f"{'='*70}")
        print(f"Total extrageri: {len(self.draws)}")
        print(f"Format: {self.config.numbers_to_draw} numere din {self.config.min_number}-{self.config.max_number}")
        if self.config.is_composite:
            print(f"  Compus:")
            for i, (count, min_val, max_val) in enumerate(self.config.composite_parts, 1):
                print(f"    Partea {i}: {count} din {min_val}-{max_val}")
        print(f"RNG types: {len(rng_types)}")
        print(f"Min matches: {min_matches}/{self.config.numbers_to_draw} ({min_matches/self.config.numbers_to_draw:.1%})")
        print(f"Success threshold: {success_threshold:.1%}")
        print(f"Search size per draw: {search_size:,}")
        print(f"Workers: {workers}")
        print()
        
        all_rng_results = {}
        
        for rng_type in rng_types:
            print(f"\n{'='*70}")
            print(f"Testing RNG: {rng_type.upper()}")
            print(f"{'='*70}\n")
            
            # Prepare tasks
            tasks = [
                (i, draw['numbers_sorted'], rng_type, self.config, seed_range, search_size, min_matches)
                for i, draw in enumerate(self.draws)
            ]
            
            start_time = time.time()
            seed_sequence = []
            success_count = 0
            
            # Parallel processing
            with Pool(processes=workers) as pool:
                for draw_idx, result in pool.imap(find_best_seed_for_draw, tasks):
                    if result and result['matches'] >= min_matches:
                        seed_sequence.append({
                            'draw_idx': draw_idx,
                            'date': self.draws[draw_idx]['date_str'],
                            'seed': result['seed'],
                            'matches': result['matches'],
                            'score': result['score']
                        })
                        success_count += 1
                        status = "✓"
                    else:
                        status = "✗"
                    
                    if (draw_idx + 1) % 50 == 0:
                        current_success_rate = success_count / (draw_idx + 1)
                        print(f"[{draw_idx+1:4}/{len(self.draws)}] Success: {success_count:3} "
                              f"({current_success_rate:.1%}) | Last: {status}")
            
            elapsed = time.time() - start_time
            success_rate = success_count / len(self.draws)
            
            print(f"\n{rng_type} Results:")
            print(f"  Success: {success_count}/{len(self.draws)} ({success_rate:.1%})")
            print(f"  Time: {elapsed:.1f}s")
            
            # Dacă success rate e peste threshold, analizează pattern
            if success_rate >= success_threshold and len(seed_sequence) >= 10:
                print(f"\n  ✓ SUCCESS RATE OVER THRESHOLD!")
                print(f"  Analyzing pattern in seed sequence...")
                
                seeds = [s['seed'] for s in seed_sequence]
                pattern_finder = AdvancedPatternFinder(seeds)
                patterns = pattern_finder.find_all_patterns()
                
                all_rng_results[rng_type] = {
                    'success_rate': success_rate,
                    'success_count': success_count,
                    'total_draws': len(self.draws),
                    'seed_sequence': seed_sequence,
                    'patterns': patterns,
                    'avg_matches': sum(s['matches'] for s in seed_sequence) / len(seed_sequence),
                }
                
                if patterns:
                    print(f"  ✓✓ PATTERNS FOUND!")
                    for p in patterns:
                        print(f"    - {p['type']}: {p['formula']} (R²={p['r_squared']:.3f})")
                else:
                    print(f"  ✗ No patterns in seeds (still good for probabilistic)")
            else:
                print(f"  ✗ Success rate too low ({success_rate:.1%} < {success_threshold:.1%})")
        
        return all_rng_results
    
    def generate_pragmatic_prediction(self, results: Dict, num_predictions: int = 5):
        """
        Generează predicții bazate pe results
        """
        print(f"\n{'='*70}")
        print(f"PRAGMATIC PREDICTIONS - {self.config.name}")
        print(f"{'='*70}\n")
        
        best_rng = None
        best_rate = 0
        
        # Găsește best RNG
        for rng_name, data in results.items():
            if data['success_rate'] > best_rate:
                best_rate = data['success_rate']
                best_rng = rng_name
        
        if not best_rng:
            print("No good RNG found for predictions!")
            return []
        
        print(f"Best RNG: {best_rng} ({best_rate:.1%} success rate)")
        
        best_data = results[best_rng]
        
        # Strategie 1: Dacă are pattern în seeds, folosește-l
        if best_data['patterns']:
            print(f"\nStrategy 1: Using PATTERN from seeds")
            best_pattern = best_data['patterns'][0]
            print(f"  Pattern: {best_pattern['formula']}")
            print(f"  Next seed: {best_pattern['next_seed']:,}")
            
            # Generează din next seed
            rng = create_rng(best_rng, best_pattern['next_seed'])
            
            if self.config.is_composite:
                prediction = []
                for count, min_val, max_val in self.config.composite_parts:
                    part_nums = generate_numbers(rng, count, min_val, max_val)
                    prediction.extend(part_nums)
            else:
                prediction = generate_numbers(rng, self.config.numbers_to_draw,
                                            self.config.min_number, self.config.max_number)
            
            return [{
                'method': 'pattern',
                'rng': best_rng,
                'seed': best_pattern['next_seed'],
                'formula': best_pattern['formula'],
                'numbers': prediction,
                'confidence': best_rate * best_pattern['r_squared']
            }]
        
        # Strategie 2: Probabilistic
        else:
            print(f"\nStrategy 2: PROBABILISTIC from seed distribution")
            
            seeds = [s['seed'] for s in best_data['seed_sequence']]
            
            seed_range_low = min(seeds)
            seed_range_high = max(seeds)
            seed_avg = int(np.mean(seeds))
            seed_median = int(np.median(seeds))
            
            print(f"  Seed range: {seed_range_low:,} - {seed_range_high:,}")
            print(f"  Seed avg: {seed_avg:,}")
            print(f"  Seed median: {seed_median:,}")
            
            predictions = []
            
            # Generează predicții din seeds tipice
            for method, seed_val in [('median_seed', seed_median), 
                                     ('average_seed', seed_avg)]:
                rng = create_rng(best_rng, seed_val)
                
                if self.config.is_composite:
                    pred = []
                    for count, min_val, max_val in self.config.composite_parts:
                        part_nums = generate_numbers(rng, count, min_val, max_val)
                        pred.extend(part_nums)
                else:
                    pred = generate_numbers(rng, self.config.numbers_to_draw,
                                          self.config.min_number, self.config.max_number)
                
                predictions.append({
                    'method': method,
                    'rng': best_rng,
                    'seed': seed_val,
                    'numbers': pred,
                    'confidence': best_rate * 0.75
                })
            
            # Recent trend
            recent_seeds = seeds[-10:]
            recent_avg = int(np.mean(recent_seeds))
            rng = create_rng(best_rng, recent_avg)
            
            if self.config.is_composite:
                pred = []
                for count, min_val, max_val in self.config.composite_parts:
                    part_nums = generate_numbers(rng, count, min_val, max_val)
                    pred.extend(part_nums)
            else:
                pred = generate_numbers(rng, self.config.numbers_to_draw,
                                      self.config.min_number, self.config.max_number)
            
            predictions.append({
                'method': 'recent_trend',
                'rng': best_rng,
                'seed': recent_avg,
                'numbers': pred,
                'confidence': best_rate * 0.7
            })
            
            return predictions[:num_predictions]


def main():
    parser = argparse.ArgumentParser(
        description='Unified Pragmatic Pattern Finder',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--lottery', type=str, required=True,
                       choices=list_available_lotteries(),
                       help='Tipul de loterie')
    parser.add_argument('--input', type=str, required=True,
                       help='Fișier JSON cu date')
    parser.add_argument('--min-matches', type=int, default=3,
                       help='Minimum matches to consider success')
    parser.add_argument('--success-threshold', type=float, default=0.65,
                       help='Min success rate to consider RNG viable (0.0-1.0)')
    parser.add_argument('--search-size', type=int, default=2000000,
                       help='Seeds to test per draw')
    parser.add_argument('--seed-range', type=int, nargs=2, default=[0, 10000000])
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--rng-types', type=str, nargs='+',
                       help='Specific RNG types to test')
    parser.add_argument('--quick-test', action='store_true',
                       help='Test only fast RNGs for quick results')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  UNIFIED PRAGMATIC PATTERN FINDER")
    print("  Finds 'good enough' patterns over LONG periods")
    print("="*70)
    
    finder = UnifiedPatternFinder(args.input, args.lottery)
    
    print(f"\n⚙️  Config:")
    print(f"  Min matches: {args.min_matches}/{finder.config.numbers_to_draw} "
          f"({args.min_matches/finder.config.numbers_to_draw:.1%})")
    print(f"  Success threshold: {args.success_threshold:.1%}")
    print(f"  Search size: {args.search_size:,} seeds/draw")
    
    # Determine RNG types to test
    if args.quick_test:
        rng_types = ['lcg_weak', 'xorshift_simple', 'lcg_glibc', 'xorshift32', 'js_math_random', 'xoshiro256']
        print(f"  Quick test mode: {len(rng_types)} fast RNGs (includes web/modern)")
    elif args.rng_types:
        rng_types = args.rng_types
    else:
        rng_types = list(RNG_TYPES.keys())
        print(f"  Testing ALL {len(rng_types)} RNG types")
    
    print()
    
    # Run analysis
    results = finder.find_pragmatic_patterns(
        rng_types=rng_types,
        seed_range=tuple(args.seed_range),
        search_size=args.search_size,
        min_matches=args.min_matches,
        success_threshold=args.success_threshold,
        workers=args.workers
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    if results:
        print(f"RNGs with success rate >= {args.success_threshold:.1%}:\n")
        
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1]['success_rate'], 
                               reverse=True)
        
        for rng_name, data in sorted_results:
            print(f"  {rng_name:20s}: {data['success_rate']:.1%} "
                  f"({data['success_count']}/{data['total_draws']}) "
                  f"| Avg: {data['avg_matches']:.1f}/{finder.config.numbers_to_draw}")
            
            if data['patterns']:
                for p in data['patterns']:
                    print(f"    └─ {p['type']}: R²={p['r_squared']:.3f}")
        
        # Generate predictions
        print(f"\n{'='*70}")
        predictions = finder.generate_pragmatic_prediction(results, num_predictions=3)
        
        if predictions:
            print(f"\n📊 GENERATED PREDICTIONS:\n")
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. Method: {pred['method']}")
                print(f"   RNG: {pred['rng']}")
                print(f"   Seed: {pred['seed']:,}")
                print(f"   Confidence: {pred['confidence']:.1%}")
                
                # Format numbers based on lottery type
                if finder.config.is_composite:
                    print(f"   🎲 Prediction:")
                    idx = 0
                    for part_i, (count, min_val, max_val) in enumerate(finder.config.composite_parts, 1):
                        part_nums = pred['numbers'][idx:idx + count]
                        print(f"      Partea {part_i} ({count} din {min_val}-{max_val}): {part_nums}")
                        idx += count
                else:
                    print(f"   🎲 Prediction: {pred['numbers']}")
                print()
        
        # Save
        output_file = f'{args.lottery}_pragmatic_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'lottery_type': args.lottery,
                'lottery_name': finder.config.name,
                'config': {
                    'min_matches': args.min_matches,
                    'success_threshold': args.success_threshold,
                    'total_draws': len(finder.draws)
                },
                'results': results,
                'predictions': predictions
            }, f, indent=2)
        
        print(f"💾 Results saved: {output_file}")
        
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print(f"{'='*70}")
        print(f"""
Success rate >= 65-70%:
  → RNG-ul poate genera consistent {args.min_matches}+/{finder.config.numbers_to_draw} matches
  → Pattern în seeds = PREDICTIBIL
  → Predicții viabile pentru test!

Success rate < 65%:
  → RNG-ul nu se potrivește consistent
  → Seeds random, fără pattern
  → NU e acest tip de RNG

Dacă NICIUN RNG >= threshold:
  → CONFIRMARE: Nu e RNG software
  → Extragere fizică aleatoare
        """)
    
    else:
        print("❌ NICIUN RNG nu atinge success threshold!")
        print(f"\nAcest lucru înseamnă:")
        print(f"  • Niciun RNG nu generează consistent {args.min_matches}+/{finder.config.numbers_to_draw} matches")
        print(f"  • Seeds variază aleatoriu, fără pattern")
        print(f"  • CONFIRMARE: Datele NU provin din RNG")
        print(f"\n  → Extragere FIZICĂ confirmată!")


if __name__ == '__main__':
    main()
