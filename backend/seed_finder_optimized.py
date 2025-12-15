#!/usr/bin/env python3
"""
Seed Finder OPTIMIZED - Versiune pentru calcule masive

Optimizări:
- Multiprocessing pentru toate CPU cores
- Batch processing eficient
- Checkpointing (salvează progres)
- Resume capability
- Memory efficient
- Progress tracking
- Rezultate incrementale

Utilizare:
    # Full power - toate cores
    python3 seed_finder_optimized.py --input loto_data.json --seed-range 0 1000000000 --workers 64
    
    # Cu checkpoint la fiecare 1M seeds
    python3 seed_finder_optimized.py --seed-range 0 1000000000 --checkpoint-every 1000000
    
    # Resume din checkpoint
    python3 seed_finder_optimized.py --resume checkpoint_12345.json
"""

import argparse
import json
import time
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count, Manager
import os
import numpy as np
from functools import partial


class FastLCG:
    """LCG optimizat pentru viteză"""
    __slots__ = ['state', 'a', 'c', 'm']
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        self.a = 1103515245
        self.c = 12345
        self.m = 2147483648  # 2^31
    
    def generate_numbers(self, count: int, min_val: int, max_val: int) -> List[int]:
        """Generează count numere unice cât mai rapid"""
        numbers = set()
        range_size = max_val - min_val + 1
        max_attempts = count * 50  # Reduce attempts pentru viteză
        
        for _ in range(max_attempts):
            self.state = (self.a * self.state + self.c) % self.m
            num = min_val + (self.state % range_size)
            numbers.add(num)
            if len(numbers) >= count:
                break
        
        return sorted(list(numbers))[:count]


class FastXorshift:
    """Xorshift optimizat"""
    __slots__ = ['state']
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1
    
    def generate_numbers(self, count: int, min_val: int, max_val: int) -> List[int]:
        numbers = set()
        range_size = max_val - min_val + 1
        max_attempts = count * 50
        
        for _ in range(max_attempts):
            x = self.state
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= (x >> 17) & 0xFFFFFFFF
            x ^= (x << 5) & 0xFFFFFFFF
            self.state = x
            
            num = min_val + (x % range_size)
            numbers.add(num)
            if len(numbers) >= count:
                break
        
        return sorted(list(numbers))[:count]


def calculate_match_score_fast(gen: List[int], actual: List[int]) -> Tuple[int, float]:
    """Calcul rapid de scor"""
    gen_set = set(gen)
    act_set = set(actual)
    matches = len(gen_set & act_set)
    return matches, matches / len(act_set)


def test_seed_worker(seed: int, target_draws: List[List[int]], 
                     rng_type: str, min_threshold: float) -> Dict:
    """Worker function pentru un seed - optimizat pentru paralelizare"""
    try:
        # Inițializare RNG
        if rng_type == 'lcg':
            rng = FastLCG(seed)
        else:
            rng = FastXorshift(seed)
        
        # Testează fiecare draw
        total_matches = 0
        total_score = 0.0
        max_matches = 0
        
        for target in target_draws:
            generated = rng.generate_numbers(6, 1, 40)
            matches, score = calculate_match_score_fast(generated, target)
            total_matches += matches
            total_score += score
            max_matches = max(max_matches, matches)
        
        avg_score = total_score / len(target_draws)
        
        # Returnează doar dacă trece threshold-ul
        if avg_score >= min_threshold:
            return {
                'seed': seed,
                'avg_score': avg_score,
                'total_matches': total_matches,
                'max_matches': max_matches,
                'avg_matches': total_matches / len(target_draws)
            }
        
        return None
    except:
        return None


def test_seed_batch(seed_batch: List[int], target_draws: List[List[int]], 
                   rng_type: str, min_threshold: float) -> List[Dict]:
    """Procesează un batch de seeds"""
    results = []
    for seed in seed_batch:
        result = test_seed_worker(seed, target_draws, rng_type, min_threshold)
        if result:
            results.append(result)
    return results


class OptimizedSeedFinder:
    def __init__(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.draws = self.data['draws']
        print(f"Încărcat {len(self.draws)} extrageri\n")
    
    def massive_search(self, 
                      seed_start: int,
                      seed_end: int,
                      target_draws: List[List[int]],
                      rng_type: str = 'lcg',
                      workers: int = None,
                      batch_size: int = 10000,
                      min_threshold: float = 0.25,
                      checkpoint_file: str = None,
                      checkpoint_every: int = 1000000) -> List[Dict]:
        """
        Căutare masivă paralelizată
        """
        if workers is None:
            workers = cpu_count()
        
        print(f"{'='*70}")
        print(f"CĂUTARE MASIVĂ PARALELIZATĂ")
        print(f"{'='*70}")
        print(f"Seed range: {seed_start:,} → {seed_end:,} ({seed_end - seed_start:,} total)")
        print(f"Workers: {workers}")
        print(f"Batch size: {batch_size:,}")
        print(f"Min threshold: {min_threshold:.1%}")
        print(f"Target draws: {len(target_draws)}")
        print(f"Checkpoint every: {checkpoint_every:,} seeds")
        print()
        
        # Verifică checkpoint existent
        start_seed = seed_start
        all_results = []
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"📁 Loading checkpoint: {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                start_seed = checkpoint_data['last_seed'] + 1
                all_results = checkpoint_data['results']
                print(f"   Resuming from seed: {start_seed:,}")
                print(f"   Already found: {len(all_results)} candidates\n")
        
        # Pregătește seed ranges pentru workers
        total_seeds = seed_end - start_seed
        seeds_per_batch = batch_size
        
        # Create batches
        seed_batches = []
        current = start_seed
        while current < seed_end:
            batch_end = min(current + seeds_per_batch, seed_end)
            seed_batches.append(list(range(current, batch_end)))
            current = batch_end
        
        print(f"Procesare {len(seed_batches):,} batches...\n")
        
        # Progress tracking
        start_time = time.time()
        processed = 0
        last_checkpoint = start_seed
        
        # Parallel processing
        worker_func = partial(test_seed_batch, 
                            target_draws=target_draws,
                            rng_type=rng_type,
                            min_threshold=min_threshold)
        
        with Pool(processes=workers) as pool:
            for i, batch_results in enumerate(pool.imap_unordered(worker_func, seed_batches)):
                # Adaugă rezultate
                all_results.extend(batch_results)
                
                # Update progress
                processed += len(seed_batches[i])
                elapsed = time.time() - start_time
                speed = processed / elapsed if elapsed > 0 else 0
                remaining = total_seeds - processed
                eta = remaining / speed if speed > 0 else 0
                
                # Progress display
                if (i + 1) % 10 == 0 or (i + 1) == len(seed_batches):
                    progress = processed / total_seeds * 100
                    print(f"\r[{progress:6.2f}%] Processed: {processed:,}/{total_seeds:,} | "
                          f"Speed: {speed:,.0f} seeds/s | Found: {len(all_results)} | "
                          f"ETA: {eta/60:.1f}m", end='', flush=True)
                
                # Checkpoint
                if checkpoint_file and (processed - last_checkpoint) >= checkpoint_every:
                    self._save_checkpoint(checkpoint_file, start_seed + processed, all_results)
                    last_checkpoint = processed
        
        print()  # New line
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"FINALIZAT")
        print(f"{'='*70}")
        print(f"Seeds procesate: {total_seeds:,}")
        print(f"Timp total: {elapsed/60:.1f} minute")
        print(f"Viteză medie: {total_seeds/elapsed:,.0f} seeds/secund")
        print(f"Candidați găsiți: {len(all_results)}")
        
        # Save final results
        if all_results:
            self._save_results(all_results, seed_start, seed_end, elapsed)
        
        # Sort by score
        all_results.sort(key=lambda x: (x['avg_score'], x['max_matches']), reverse=True)
        
        return all_results
    
    def _save_checkpoint(self, filename: str, last_seed: int, results: List[Dict]):
        """Salvează checkpoint"""
        checkpoint_data = {
            'last_seed': last_seed,
            'results': results,
            'timestamp': time.time()
        }
        with open(filename, 'w') as f:
            json.dump(checkpoint_data, f)
    
    def _save_results(self, results: List[Dict], seed_start: int, seed_end: int, elapsed: float):
        """Salvează rezultate finale"""
        output_file = f"seed_results_{seed_start}_{seed_end}_{int(time.time())}.json"
        
        output_data = {
            'seed_range': [seed_start, seed_end],
            'total_seeds_tested': seed_end - seed_start,
            'execution_time_seconds': elapsed,
            'candidates_found': len(results),
            'timestamp': time.time(),
            'results': results[:1000]  # Top 1000
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Rezultate salvate: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Seed Finder OPTIMIZED - Pentru calcule masive',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemple:
  # Căutare pe range mare cu toate cores
  python3 seed_finder_optimized.py --seed-range 0 100000000 --workers 64
  
  # Cu checkpoint automat
  python3 seed_finder_optimized.py --seed-range 0 1000000000 --checkpoint checkpoint.json --checkpoint-every 10000000
  
  # Resume din checkpoint
  python3 seed_finder_optimized.py --resume checkpoint.json
        """
    )
    
    parser.add_argument('--input', type=str, default='/app/backend/loto_data.json')
    parser.add_argument('--seed-range', type=int, nargs=2, metavar=('START', 'END'),
                       help='Range de seeds de testat (ex: 0 1000000000)')
    parser.add_argument('--draws', type=int, default=2,
                       help='Număr extrageri consecutive')
    parser.add_argument('--workers', type=int, default=None,
                       help='Număr de workers (default: toate CPU cores)')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Seeds per batch')
    parser.add_argument('--threshold', type=float, default=0.25,
                       help='Threshold minim pentru rezultate (0.0-1.0)')
    parser.add_argument('--rng', type=str, choices=['lcg', 'xorshift'], default='lcg')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--checkpoint-every', type=int, default=1000000,
                       help='Seeds între checkpoints')
    parser.add_argument('--resume', type=str, help='Resume din checkpoint')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  SEED FINDER OPTIMIZED - High Performance Computing")
    print("="*70)
    print()
    
    # CPU info
    available_cores = cpu_count()
    workers = args.workers if args.workers else available_cores
    print(f"🖥️  CPU Cores disponibile: {available_cores}")
    print(f"👷 Workers configurați: {workers}")
    print()
    
    try:
        finder = OptimizedSeedFinder(args.input)
        
        # Determină target draws
        target_draws = [finder.draws[i]['numbers_sorted'] 
                       for i in range(min(args.draws, len(finder.draws)))]
        
        print(f"🎯 Target extrageri:")
        for i, draw in enumerate(target_draws):
            print(f"   {i+1}. {finder.draws[i]['date_str']}: {draw}")
        print()
        
        # Determină seed range
        if args.resume:
            # Resume din checkpoint
            results = finder.massive_search(
                seed_start=0,  # Will be overridden by checkpoint
                seed_end=4294967296,  # 2^32
                target_draws=target_draws,
                rng_type=args.rng,
                workers=workers,
                batch_size=args.batch_size,
                min_threshold=args.threshold,
                checkpoint_file=args.resume,
                checkpoint_every=args.checkpoint_every
            )
        elif args.seed_range:
            results = finder.massive_search(
                seed_start=args.seed_range[0],
                seed_end=args.seed_range[1],
                target_draws=target_draws,
                rng_type=args.rng,
                workers=workers,
                batch_size=args.batch_size,
                min_threshold=args.threshold,
                checkpoint_file=args.checkpoint,
                checkpoint_every=args.checkpoint_every
            )
        else:
            print("Eroare: Specifică --seed-range sau --resume")
            return
        
        # Display top results
        if results:
            print(f"\n{'='*70}")
            print(f"TOP 20 REZULTATE")
            print(f"{'='*70}\n")
            print(f"{'Rank':<6} {'Seed':<15} {'Avg Score':<12} {'Avg Match':<12} {'Max Match'}")
            print("-" * 70)
            
            for i, r in enumerate(results[:20], 1):
                print(f"{i:<6} {r['seed']:<15,} {r['avg_score']:<12.2%} "
                      f"{r['avg_matches']:<12.2f} {r['max_matches']}/6")
        else:
            print("\nNu s-au găsit seed-uri peste threshold.")
            print(f"Încercați să reduceți --threshold (actual: {args.threshold})")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Întrerupt de utilizator")
        if args.checkpoint:
            print(f"   Progress salvat în: {args.checkpoint}")
            print(f"   Rulează cu --resume {args.checkpoint} pentru a continua")
    except Exception as e:
        print(f"\nEroare: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
