#!/usr/bin/env python3
"""
Pragmatic Pattern Finder - Găsește "good enough" patterns pe termen LUNG

NU caută 100% perfect - caută pattern care nimerește ~70-80% decent!

Abordare:
1. Testează pe 10-20 ani (500-2000 extrageri)
2. Pentru fiecare RNG, găsește seeds cu 3+/6 matches
3. Analizează dacă există pattern în seeds "good enough"
4. Success rate: % extrageri cu >=3 matches

Utilizare:
    # Test pe toată arhiva
    python3 pragmatic_pattern_finder.py --years all --min-matches 3

    # Test pe ultimii 10 ani
    python3 pragmatic_pattern_finder.py --years 10 --min-matches 3 --success-threshold 0.70
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


def evaluate_seed_quality(seed: int, target: List[int], rng_type: str) -> Dict:
    """Evaluează calitatea unui seed pentru o extragere"""
    try:
        rng = create_rng(rng_type, seed)
        generated = generate_numbers(rng, 6, 1, 40)
        matches = len(set(generated) & set(target))
        
        return {
            'seed': seed,
            'matches': matches,
            'generated': generated,
            'score': matches / 6.0
        }
    except:
        return None


def find_best_seed_for_draw(args):
    """Pentru o extragere, găsește BEST seed pentru un RNG"""
    draw_idx, target, rng_type, seed_range, search_size, min_matches = args
    
    import random
    test_seeds = random.sample(range(seed_range[0], seed_range[1]),
                               min(search_size, seed_range[1] - seed_range[0]))
    
    best_result = None
    best_matches = 0
    
    target_set = set(target)
    
    for seed in test_seeds:
        try:
            rng = create_rng(rng_type, seed)
            generated = generate_numbers(rng, 6, 1, 40)
            matches = len(set(generated) & target_set)
            
            if matches >= min_matches and matches > best_matches:
                best_matches = matches
                best_result = {
                    'seed': seed,
                    'matches': matches,
                    'generated': generated,
                    'score': matches / 6.0
                }
                
                if matches == 6:  # Perfect!
                    break
        except:
            continue
    
    return draw_idx, best_result


class PragmaticPatternFinder:
    def __init__(self, data_file: str):
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.draws = data['draws']
        print(f"Încărcat {len(self.draws)} extrageri\n")
    
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
        print(f"PRAGMATIC PATTERN FINDER")
        print(f"{'='*70}")
        print(f"Total extrageri: {len(self.draws)}")
        print(f"RNG types: {len(rng_types)}")
        print(f"Min matches: {min_matches}/6 ({min_matches/6:.1%})")
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
                (i, draw['numbers_sorted'], rng_type, seed_range, search_size, min_matches)
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
        print(f"PRAGMATIC PREDICTIONS")
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
            prediction = generate_numbers(rng, 6, 1, 40)
            
            return [{
                'method': 'pattern',
                'rng': best_rng,
                'seed': best_pattern['next_seed'],
                'formula': best_pattern['formula'],
                'numbers': prediction,
                'confidence': best_rate * best_pattern['r_squared']
            }]
        
        # Strategie 2: Probabilistic - ia cele mai frecvente seeds
        else:
            print(f"\nStrategy 2: PROBABILISTIC from seed distribution")
            
            seeds = [s['seed'] for s in best_data['seed_sequence']]
            
            # Analizează distributia seeds
            seed_range_low = min(seeds)
            seed_range_high = max(seeds)
            seed_avg = int(np.mean(seeds))
            seed_median = int(np.median(seeds))
            
            print(f"  Seed range: {seed_range_low:,} - {seed_range_high:,}")
            print(f"  Seed avg: {seed_avg:,}")
            print(f"  Seed median: {seed_median:,}")
            
            # Generează predicții din seeds "tipice"
            predictions = []
            
            # 1. Din median
            rng = create_rng(best_rng, seed_median)
            pred1 = generate_numbers(rng, 6, 1, 40)
            predictions.append({
                'method': 'median_seed',
                'rng': best_rng,
                'seed': seed_median,
                'numbers': pred1,
                'confidence': best_rate * 0.8
            })
            
            # 2. Din average
            rng = create_rng(best_rng, seed_avg)
            pred2 = generate_numbers(rng, 6, 1, 40)
            predictions.append({
                'method': 'average_seed',
                'rng': best_rng,
                'seed': seed_avg,
                'numbers': pred2,
                'confidence': best_rate * 0.7
            })
            
            # 3. Din ultimele seeds (trend recent)
            recent_seeds = seeds[-10:]
            recent_avg = int(np.mean(recent_seeds))
            rng = create_rng(best_rng, recent_avg)
            pred3 = generate_numbers(rng, 6, 1, 40)
            predictions.append({
                'method': 'recent_trend',
                'rng': best_rng,
                'seed': recent_avg,
                'numbers': pred3,
                'confidence': best_rate * 0.75
            })
            
            return predictions[:num_predictions]


def main():
    parser = argparse.ArgumentParser(
        description='Pragmatic Pattern Finder - Good enough patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', type=str, default='/app/backend/loto_data.json')
    parser.add_argument('--years', type=str, default='all',
                       help='How many years: all, 10, 20, etc')
    parser.add_argument('--min-matches', type=int, default=3,
                       help='Minimum matches to consider success (3-6)')
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
    print("  PRAGMATIC PATTERN FINDER")
    print("  Finds 'good enough' patterns over LONG periods")
    print("="*70)
    print(f"\n⚙️  Config:")
    print(f"  Min matches: {args.min_matches}/6 ({args.min_matches/6:.1%})")
    print(f"  Success threshold: {args.success_threshold:.1%}")
    print(f"  Search size: {args.search_size:,} seeds/draw")
    
    finder = PragmaticPatternFinder(args.input)
    
    # Determine RNG types to test
    if args.quick_test:
        rng_types = ['lcg_weak', 'xorshift_simple', 'lcg_glibc', 'xorshift32']
        print(f"  Quick test mode: {len(rng_types)} fast RNGs")
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
                  f"| Avg: {data['avg_matches']:.1f}/6")
            
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
                print(f"   🎲 Prediction: {pred['numbers']}")
                print()
        
        # Save
        output_file = 'pragmatic_results.json'
        with open(output_file, 'w') as f:
            json.dump({
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
        print("""
Success rate >= 65-70%:
  → RNG-ul poate genera consistent 3+/6 matches
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
        print(f"  • Niciun RNG nu generează consistent {args.min_matches}+/6 matches")
        print(f"  • Seeds variază aleatoriu, fără pattern")
        print(f"  • CONFIRMARE: Datele NU provin din RNG")
        print(f"\n  → Extragere FIZICĂ confirmată!")


if __name__ == '__main__':
    main()
