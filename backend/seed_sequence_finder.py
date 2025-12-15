#!/usr/bin/env python3
"""
Seed Sequence Finder - Găsește seed-ul pentru FIECARE extragere individuală

Acest script găsește ce seed RNG a fost folosit pentru fiecare extragere din istoric.
Rezultă o SECVENȚĂ de seeds: [S₁, S₂, S₃, ...]

Utilizare:
    python3 seed_sequence_finder.py --input loto_data.json --output seed_sequence.json
    python3 seed_sequence_finder.py --input loto_data.json --start 0 --end 50
"""

import argparse
import json
import time
from typing import List, Dict, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np


class FastLCG:
    """LCG optimizat"""
    __slots__ = ['state', 'a', 'c', 'm']
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        self.a = 1103515245
        self.c = 12345
        self.m = 2147483648
    
    def generate_numbers(self, count: int, min_val: int, max_val: int) -> List[int]:
        numbers = set()
        range_size = max_val - min_val + 1
        
        for _ in range(count * 50):
            self.state = (self.a * self.state + self.c) % self.m
            num = min_val + (self.state % range_size)
            numbers.add(num)
            if len(numbers) >= count:
                break
        
        return sorted(list(numbers))[:count]


def find_seed_for_draw(target: List[int], seed_range: tuple, rng_type: str = 'lcg') -> Optional[Dict]:
    """
    Găsește seed-ul care generează exact target
    Returnează primul seed găsit + scor
    """
    seed_start, seed_end = seed_range
    target_set = set(target)
    
    best_seed = None
    best_matches = 0
    
    for seed in range(seed_start, seed_end):
        rng = FastLCG(seed)
        generated = rng.generate_numbers(6, 1, 40)
        
        matches = len(set(generated) & target_set)
        
        if matches == 6:  # Perfect match!
            return {
                'seed': seed,
                'matches': 6,
                'generated': generated,
                'perfect': True
            }
        
        if matches > best_matches:
            best_matches = matches
            best_seed = seed
            best_generated = generated
    
    # Returnează cel mai bun găsit
    if best_seed is not None:
        return {
            'seed': best_seed,
            'matches': best_matches,
            'generated': best_generated,
            'perfect': False
        }
    
    return None


def find_seed_worker(args):
    """Worker pentru găsirea seed-ului unei extrageri"""
    draw_idx, draw_data, seed_range, rng_type, search_size = args
    
    target = draw_data['numbers_sorted']
    
    # Sample random din seed_range
    full_start, full_end = seed_range
    
    # Pentru fiecare draw, testăm un sample random
    import random
    test_seeds = random.sample(range(full_start, full_end), 
                              min(search_size, full_end - full_start))
    
    best_result = None
    best_matches = 0
    
    target_set = set(target)
    
    for seed in test_seeds:
        rng = FastLCG(seed)
        generated = rng.generate_numbers(6, 1, 40)
        matches = len(set(generated) & target_set)
        
        if matches == 6:  # Perfect!
            return {
                'draw_idx': draw_idx,
                'date': draw_data['date_str'],
                'target': target,
                'seed': seed,
                'matches': 6,
                'generated': generated,
                'perfect': True
            }
        
        if matches > best_matches:
            best_matches = matches
            best_result = {
                'draw_idx': draw_idx,
                'date': draw_data['date_str'],
                'target': target,
                'seed': seed,
                'matches': matches,
                'generated': generated,
                'perfect': False
            }
    
    return best_result


class SeedSequenceFinder:
    def __init__(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.draws = self.data['draws']
        print(f"Încărcat {len(self.draws)} extrageri\n")
    
    def find_sequence(self, 
                     start_idx: int = 0,
                     end_idx: int = None,
                     seed_range: tuple = (0, 10000000),
                     search_size: int = 1000000,
                     workers: int = None,
                     rng_type: str = 'lcg') -> List[Dict]:
        """
        Găsește seed-ul pentru fiecare extragere
        """
        if end_idx is None:
            end_idx = len(self.draws)
        
        if workers is None:
            workers = cpu_count()
        
        draws_to_process = self.draws[start_idx:end_idx]
        
        print(f"{'='*70}")
        print(f"SEED SEQUENCE FINDER")
        print(f"{'='*70}")
        print(f"Extrageri: {start_idx} → {end_idx-1} ({len(draws_to_process)} total)")
        print(f"Seed range: {seed_range[0]:,} → {seed_range[1]:,}")
        print(f"Search size per draw: {search_size:,} seeds")
        print(f"Workers: {workers}")
        print()
        
        # Pregătește task-uri
        tasks = [
            (start_idx + i, draw, seed_range, rng_type, search_size)
            for i, draw in enumerate(draws_to_process)
        ]
        
        print(f"Procesare {len(tasks)} extrageri...\n")
        
        start_time = time.time()
        results = []
        
        # Parallel processing
        with Pool(processes=workers) as pool:
            for i, result in enumerate(pool.imap(find_seed_worker, tasks)):
                if result:
                    results.append(result)
                    
                    status = "✓ PERFECT" if result['perfect'] else f"✗ {result['matches']}/6"
                    print(f"[{i+1:3}/{len(tasks)}] {result['date']}: Seed {result['seed']:>10,} | {status}")
                else:
                    print(f"[{i+1:3}/{len(tasks)}] FAILED - no seed found")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"COMPLETE")
        print(f"{'='*70}")
        print(f"Timp: {elapsed:.1f} secunde")
        print(f"Seeds găsite: {len(results)}/{len(tasks)}")
        
        # Statistici
        perfect_count = sum(1 for r in results if r['perfect'])
        if results:
            avg_matches = sum(r['matches'] for r in results) / len(results)
            print(f"Perfect matches: {perfect_count}")
            print(f"Average matches: {avg_matches:.2f}/6")
        
        return results
    
    def save_sequence(self, results: List[Dict], output_file: str):
        """Salvează secvența de seeds"""
        output_data = {
            'total_draws': len(results),
            'perfect_matches': sum(1 for r in results if r['perfect']),
            'timestamp': time.time(),
            'seed_sequence': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Seed sequence salvată: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Seed Sequence Finder - Găsește seed pentru fiecare extragere'
    )
    parser.add_argument('--input', type=str, default='/app/backend/loto_data.json')
    parser.add_argument('--output', type=str, default='seed_sequence.json')
    parser.add_argument('--start', type=int, default=0, help='Index start extragere')
    parser.add_argument('--end', type=int, default=None, help='Index end extragere')
    parser.add_argument('--seed-range', type=int, nargs=2, default=[0, 10000000],
                       metavar=('START', 'END'))
    parser.add_argument('--search-size', type=int, default=1000000,
                       help='Seeds de testat per extragere')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--rng', type=str, default='lcg', choices=['lcg', 'xorshift'])
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  SEED SEQUENCE FINDER")
    print("="*70)
    print("\n⚠️  Acest script găsește ce seed ar fi fost folosit pentru")
    print("    fiecare extragere (dacă ar fi fost RNG).\n")
    
    try:
        finder = SeedSequenceFinder(args.input)
        
        results = finder.find_sequence(
            start_idx=args.start,
            end_idx=args.end,
            seed_range=tuple(args.seed_range),
            search_size=args.search_size,
            workers=args.workers,
            rng_type=args.rng
        )
        
        if results:
            finder.save_sequence(results, args.output)
            
            print("\n" + "="*70)
            print("NEXT STEP")
            print("="*70)
            print(f"\nAcum analizează secvența cu:")
            print(f"  python3 seed_pattern_analyzer.py --input {args.output}")
            print("\nAcesta va căuta pattern-ul în secvența de seeds!")
        else:
            print("\nNu s-au găsit seeds pentru nicio extragere.")
    
    except Exception as e:
        print(f"Eroare: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
