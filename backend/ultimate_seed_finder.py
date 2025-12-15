#!/usr/bin/env python3
"""
ULTIMATE Seed Finder - Testează TOATE RNG-urile + TOATE Pattern-urile

Pentru FIECARE extragere, testează:
- 13 tipuri diferite de RNG-uri
- Găsește best-match seed pentru fiecare RNG
- Pattern analyzer cu 10+ formule matematice

ULTRA-OPTIMIZAT pentru mașinării puternice!

Utilizare:
    python3 ultimate_seed_finder.py --input loto_data.json --end 50 --search-size 5000000 --workers 32
"""

import argparse
import json
import time
from typing import List, Dict
from multiprocessing import Pool, cpu_count
import sys

# Import RNG library
from advanced_rng_library import RNG_TYPES, create_rng, generate_numbers
from advanced_pattern_finder import AdvancedPatternFinder


def find_best_seed_all_rngs(args):
    """
    Pentru o extragere, găsește best seed pentru FIECARE tip de RNG
    """
    draw_idx, target, seed_range, search_size = args
    
    results_per_rng = {}
    target_set = set(target)
    
    # Testează fiecare tip de RNG
    for rng_name in RNG_TYPES.keys():
        best_seed = None
        best_matches = 0
        best_generated = None
        
        # Sample random seeds
        import random
        test_seeds = random.sample(range(seed_range[0], seed_range[1]), 
                                   min(search_size, seed_range[1] - seed_range[0]))
        
        for seed in test_seeds:
            try:
                rng = create_rng(rng_name, seed)
                generated = generate_numbers(rng, 6, 1, 40)
                matches = len(set(generated) & target_set)
                
                if matches == 6:  # Perfect!
                    results_per_rng[rng_name] = {
                        'seed': seed,
                        'matches': 6,
                        'generated': generated,
                        'perfect': True
                    }
                    break
                
                if matches > best_matches:
                    best_matches = matches
                    best_seed = seed
                    best_generated = generated
            except:
                continue
        
        if rng_name not in results_per_rng and best_seed is not None:
            results_per_rng[rng_name] = {
                'seed': best_seed,
                'matches': best_matches,
                'generated': best_generated,
                'perfect': False
            }
    
    return draw_idx, results_per_rng


class UltimateSeedFinder:
    def __init__(self, data_file: str):
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.draws = data['draws']
        print(f"Încărcat {len(self.draws)} extrageri\n")
    
    def find_ultimate_sequence(self,
                               start_idx: int = 0,
                               end_idx: int = None,
                               seed_range: tuple = (0, 10000000),
                               search_size: int = 1000000,
                               workers: int = None):
        """
        Găsește seed pentru FIECARE extragere cu FIECARE RNG
        """
        if end_idx is None:
            end_idx = len(self.draws)
        
        if workers is None:
            workers = cpu_count()
        
        draws_subset = self.draws[start_idx:end_idx]
        
        print(f"{'='*70}")
        print(f"ULTIMATE SEED FINDER")
        print(f"{'='*70}")
        print(f"Extrageri: {start_idx} → {end_idx-1} ({len(draws_subset)} total)")
        print(f"RNG types: {len(RNG_TYPES)} different algorithms")
        print(f"Search size per RNG: {search_size:,} seeds")
        print(f"Total tests: {len(draws_subset) * len(RNG_TYPES) * search_size:,}")
        print(f"Workers: {workers}")
        print()
        
        # Prepare tasks
        tasks = [
            (start_idx + i, draw['numbers_sorted'], seed_range, search_size)
            for i, draw in enumerate(draws_subset)
        ]
        
        print(f"Processing {len(tasks)} draws with {len(RNG_TYPES)} RNG types each...")
        print(f"Total RNG-draw combinations: {len(tasks) * len(RNG_TYPES)}\n")
        
        start_time = time.time()
        all_results = {}
        
        # Parallel processing
        with Pool(processes=workers) as pool:
            for draw_idx, rng_results in pool.imap_unordered(find_best_seed_all_rngs, tasks):
                all_results[draw_idx] = {
                    'draw_idx': draw_idx,
                    'date': self.draws[draw_idx]['date_str'],
                    'target': self.draws[draw_idx]['numbers_sorted'],
                    'rng_results': rng_results
                }
                
                # Print progress
                best_overall = max(rng_results.values(), key=lambda x: x['matches'])
                best_rng_name = [k for k, v in rng_results.items() if v['matches'] == best_overall['matches']][0]
                
                print(f"[{len(all_results):3}/{len(tasks)}] {self.draws[draw_idx]['date_str']}: "
                      f"Best = {best_overall['matches']}/6 ({best_rng_name})")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"COMPLETE")
        print(f"{'='*70}")
        print(f"Timp: {elapsed:.1f} secunde ({elapsed/60:.1f} minute)")
        print(f"Vitez\u0103: {len(tasks) * len(RNG_TYPES) * search_size / elapsed:,.0f} tests/sec")
        
        # Analyze results
        self._analyze_results(all_results)
        
        # Save
        output_file = f'ultimate_seeds_{start_idx}_{end_idx}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'draws': len(all_results),
                'rng_types': list(RNG_TYPES.keys()),
                'results': all_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved: {output_file}")
        
        return all_results
    
    def _analyze_results(self, results: Dict):
        """Analizează ce RNG are cele mai bune rezultate"""
        print(f"\n{'='*70}")
        print(f"ANALYSIS - BEST RNG PER DRAW")
        print(f"{'='*70}\n")
        
        rng_wins = {rng: 0 for rng in RNG_TYPES.keys()}
        rng_total_matches = {rng: [] for rng in RNG_TYPES.keys()}
        
        for draw_data in results.values():
            best_match = 0
            best_rng = None
            
            for rng_name, rng_result in draw_data['rng_results'].items():
                matches = rng_result['matches']
                rng_total_matches[rng_name].append(matches)
                
                if matches > best_match:
                    best_match = matches
                    best_rng = rng_name
            
            if best_rng:
                rng_wins[best_rng] += 1
        
        # Sort by wins
        sorted_rngs = sorted(rng_wins.items(), key=lambda x: x[1], reverse=True)
        
        print("Best RNG by number of 'wins' (highest match per draw):")
        for rng_name, wins in sorted_rngs[:10]:
            avg_matches = sum(rng_total_matches[rng_name]) / len(rng_total_matches[rng_name])
            print(f"  {rng_name:15s}: {wins:3d} wins | Avg: {avg_matches:.2f}/6 matches")
        
        # Overall stats
        print(f"\nOverall Statistics:")
        for rng_name in sorted_rngs[:5]:
            matches_list = rng_total_matches[rng_name[0]]
            print(f"  {rng_name[0]:15s}: Min={min(matches_list)} Max={max(matches_list)} "
                  f"Avg={sum(matches_list)/len(matches_list):.2f}")


def run_pattern_analysis(results_file: str, output_file: str = 'ultimate_patterns.json'):
    """
    Analizează pattern-uri pentru FIECARE RNG separat
    """
    print(f"\n{'='*70}")
    print(f"PATTERN ANALYSIS - ALL RNGs")
    print(f"{'='*70}\n")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    rng_types = data['rng_types']
    
    all_patterns = {}
    
    for rng_name in rng_types:
        print(f"\nAnalyzing {rng_name}...")
        print("-" * 70)
        
        # Extract seed sequence for this RNG
        seed_sequence = []
        for draw_idx in sorted([int(k) for k in results.keys()]):
            draw_data = results[str(draw_idx)]
            if rng_name in draw_data['rng_results']:
                seed_sequence.append(draw_data['rng_results'][rng_name]['seed'])
        
        if len(seed_sequence) < 3:
            print(f"  Insufficient seeds for {rng_name}")
            continue
        
        # Run pattern finder
        finder = AdvancedPatternFinder(seed_sequence)
        patterns = finder.find_all_patterns()
        
        if patterns:
            all_patterns[rng_name] = patterns
            print(f"  ✓ Found {len(patterns)} pattern(s) for {rng_name}")
            for p in patterns:
                print(f"    - {p['type']}: R²={p['r_squared']:.3f}")
        else:
            print(f"  ✗ No patterns found for {rng_name}")
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(all_patterns, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"PATTERN ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal RNGs with patterns: {len(all_patterns)}/{len(rng_types)}")
    print(f"💾 Patterns saved: {output_file}")
    
    return all_patterns


def main():
    parser = argparse.ArgumentParser(
        description='ULTIMATE Seed Finder - Tests ALL RNG types',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', type=str, default='/app/backend/loto_data.json')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=20)
    parser.add_argument('--seed-range', type=int, nargs=2, default=[0, 10000000])
    parser.add_argument('--search-size', type=int, default=1000000,
                       help='Seeds to test PER RNG type')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--analyze-only', type=str,
                       help='Skip seed finding, only analyze existing results file')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  ULTIMATE SEED FINDER")
    print("  Tests ALL 13 RNG Types + ALL Pattern Formulas")
    print("="*70)
    
    if args.analyze_only:
        # Just analyze existing results
        run_pattern_analysis(args.analyze_only)
    else:
        # Full workflow
        finder = UltimateSeedFinder(args.input)
        
        results = finder.find_ultimate_sequence(
            start_idx=args.start,
            end_idx=args.end,
            seed_range=tuple(args.seed_range),
            search_size=args.search_size,
            workers=args.workers
        )
        
        # Pattern analysis
        results_file = f'ultimate_seeds_{args.start}_{args.end}.json'
        patterns = run_pattern_analysis(results_file)
        
        if patterns:
            print(f"\n{'='*70}")
            print("SUMMARY - RNGs WITH PATTERNS")
            print(f"{'='*70}\n")
            
            for rng_name, pattern_list in patterns.items():
                print(f"{rng_name}:")
                for p in pattern_list:
                    print(f"  ✓ {p['type']}: {p['formula']}")
                    print(f"    Next seed: {p['next_seed']:,}")
        else:
            print(f"\n{'='*70}")
            print("NO PATTERNS FOUND IN ANY RNG")
            print(f"{'='*70}")
            print("\nThis confirms: Data does NOT come from any known RNG!")


if __name__ == '__main__':
    main()
