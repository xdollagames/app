#!/usr/bin/env python3
"""
Seed Finder GPU - Versiune cu accelerare GPU (CUDA)

Necesită:
- NVIDIA GPU cu CUDA
- numba cu suport CUDA: pip install numba
- cupy (opțional): pip install cupy-cuda11x

Utilizare:
    python3 seed_finder_gpu.py --seed-range 0 100000000 --gpu-batch 1000000

Note:
- Mult mai rapid decât CPU pentru calcule matematice simple
- Ideal pentru testarea a miliarde de seeds
- Necesită GPU cu VRAM suficient
"""

import argparse
import json
import time
import numpy as np
from typing import List, Dict

try:
    from numba import cuda, uint32, int32
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️  Numba cu CUDA nu este disponibil")
    print("   Instalare: pip install numba")


if CUDA_AVAILABLE:
    @cuda.jit
    def test_seeds_lcg_kernel(seeds, target_flat, results, num_targets, threshold):
        """
        CUDA kernel pentru testare seeds cu LCG
        
        Args:
            seeds: Array de seeds de testat
            target_flat: Array cu toate targets (flatten)
            results: Array pentru rezultate [seed_idx, score]
            num_targets: Număr de target draws
            threshold: Threshold minim
        """
        idx = cuda.grid(1)
        
        if idx < seeds.shape[0]:
            seed = seeds[idx]
            
            # LCG parameters
            a = uint32(1103515245)
            c = uint32(12345)
            m = uint32(2147483648)
            
            state = uint32(seed)
            total_matches = 0
            
            # Pentru fiecare target draw
            for target_idx in range(num_targets):
                # Generează 6 numere
                generated = cuda.local.array(6, dtype=int32)
                gen_count = 0
                attempts = 0
                
                while gen_count < 6 and attempts < 300:
                    state = (a * state + c) % m
                    num = int32(1 + (state % 40))
                    
                    # Check if unique
                    is_unique = True
                    for i in range(gen_count):
                        if generated[i] == num:
                            is_unique = False
                            break
                    
                    if is_unique:
                        generated[gen_count] = num
                        gen_count += 1
                    
                    attempts += 1
                
                # Count matches cu target
                target_offset = target_idx * 6
                matches = 0
                
                for i in range(6):
                    for j in range(6):
                        if generated[i] == target_flat[target_offset + j]:
                            matches += 1
                            break
                
                total_matches += matches
            
            # Calculează scor mediu
            avg_score = float(total_matches) / float(num_targets * 6)
            
            # Salvează dacă peste threshold
            if avg_score >= threshold:
                results[idx, 0] = seed
                results[idx, 1] = avg_score
            else:
                results[idx, 0] = -1
                results[idx, 1] = 0.0


class GPUSeedFinder:
    def __init__(self, data_file: str):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA nu este disponibil pe acest sistem")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.draws = self.data['draws']
        
        # GPU info
        gpu = cuda.get_current_device()
        print(f"\n🎮 GPU detectat: {gpu.name}")
        print(f"   Compute Capability: {gpu.compute_capability}")
        print(f"   Total Memory: {gpu.TOTAL_MEMORY / 1024**3:.1f} GB")
        print(f"   Multiprocessors: {gpu.MULTIPROCESSOR_COUNT}")
        print()
    
    def gpu_massive_search(self,
                          seed_start: int,
                          seed_end: int,
                          target_draws: List[List[int]],
                          gpu_batch_size: int = 1000000,
                          threshold: float = 0.25) -> List[Dict]:
        """
        Căutare masivă pe GPU
        """
        print(f"{'='*70}")
        print(f"CĂUTARE MASIVĂ GPU (CUDA)")
        print(f"{'='*70}")
        print(f"Seed range: {seed_start:,} → {seed_end:,}")
        print(f"GPU batch size: {gpu_batch_size:,}")
        print(f"Target draws: {len(target_draws)}")
        print(f"Threshold: {threshold:.1%}")
        print()
        
        # Pregătește target data
        target_flat = np.array([num for draw in target_draws for num in draw], dtype=np.int32)
        
        all_results = []
        total_seeds = seed_end - seed_start
        processed = 0
        start_time = time.time()
        
        # Procesează în batch-uri
        current_seed = seed_start
        
        while current_seed < seed_end:
            batch_end = min(current_seed + gpu_batch_size, seed_end)
            batch_size = batch_end - current_seed
            
            # Pregătește batch
            seeds = np.arange(current_seed, batch_end, dtype=np.uint32)
            results = np.zeros((batch_size, 2), dtype=np.float32)
            
            # Transfer la GPU
            d_seeds = cuda.to_device(seeds)
            d_target = cuda.to_device(target_flat)
            d_results = cuda.to_device(results)
            
            # Configure kernel
            threads_per_block = 256
            blocks = (batch_size + threads_per_block - 1) // threads_per_block
            
            # Execute kernel
            test_seeds_lcg_kernel[blocks, threads_per_block](
                d_seeds, d_target, d_results, len(target_draws), threshold
            )
            
            # Get results
            results = d_results.copy_to_host()
            
            # Extract valid results
            for i in range(batch_size):
                if results[i, 0] >= 0:  # Valid result
                    all_results.append({
                        'seed': int(results[i, 0]),
                        'avg_score': float(results[i, 1]),
                        'total_matches': int(results[i, 1] * len(target_draws) * 6),
                        'avg_matches': results[i, 1] * 6
                    })
            
            # Progress
            processed += batch_size
            elapsed = time.time() - start_time
            speed = processed / elapsed
            remaining = total_seeds - processed
            eta = remaining / speed if speed > 0 else 0
            
            progress = processed / total_seeds * 100
            print(f"\r[{progress:6.2f}%] {processed:,}/{total_seeds:,} | "
                  f"{speed:,.0f} seeds/s | Found: {len(all_results)} | "
                  f"ETA: {eta/60:.1f}m", end='', flush=True)
            
            current_seed = batch_end
        
        print()
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"FINALIZAT")
        print(f"{'='*70}")
        print(f"Seeds procesate: {total_seeds:,}")
        print(f"Timp: {elapsed/60:.1f} minute")
        print(f"Viteză: {total_seeds/elapsed:,.0f} seeds/sec")
        print(f"Găsite: {len(all_results)} candidați")
        
        # Sort
        all_results.sort(key=lambda x: x['avg_score'], reverse=True)
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Seed Finder GPU - CUDA accelerated'
    )
    parser.add_argument('--input', type=str, default='/app/backend/loto_data.json')
    parser.add_argument('--seed-range', type=int, nargs=2, required=True, 
                       metavar=('START', 'END'))
    parser.add_argument('--draws', type=int, default=2)
    parser.add_argument('--gpu-batch', type=int, default=1000000,
                       help='Seeds per GPU batch')
    parser.add_argument('--threshold', type=float, default=0.25)
    
    args = parser.parse_args()
    
    if not CUDA_AVAILABLE:
        print("\nCUDA nu este disponibil!")
        print("Verificați:")
        print("  1. NVIDIA GPU este prezent")
        print("  2. CUDA toolkit instalat")
        print("  3. numba instalat: pip install numba")
        return
    
    print("\n" + "="*70)
    print("  SEED FINDER GPU - CUDA Acceleration")
    print("="*70)
    
    try:
        finder = GPUSeedFinder(args.input)
        
        target_draws = [finder.draws[i]['numbers_sorted']
                       for i in range(min(args.draws, len(finder.draws)))]
        
        results = finder.gpu_massive_search(
            seed_start=args.seed_range[0],
            seed_end=args.seed_range[1],
            target_draws=target_draws,
            gpu_batch_size=args.gpu_batch,
            threshold=args.threshold
        )
        
        if results:
            print(f"\nTOP 20:")
            for i, r in enumerate(results[:20], 1):
                print(f"{i:2}. Seed {r['seed']:<12,}: {r['avg_score']:.2%} "
                      f"({r['avg_matches']:.1f}/6 avg)")
    
    except Exception as e:
        print(f"Eroare: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
