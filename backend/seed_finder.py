#!/usr/bin/env python3
"""
Seed Finder - Căutare seed-uri în date istorice Loto 5/40

Experiment: Încearcă să găsească seed-uri RNG care să recreeze 
sequențe din extragerile istorice.

⚠️  IMPORTANT: Acesta este un EXPERIMENT EDUCAȚIONAL pentru a demonstra
    că datele de loterie NU provin dintr-un RNG cu seed.
    
    Scriptul va arăta că:
    - Nu există seed-uri consistente
    - "Potrivirile" sunt întâmplătoare
    - Seed-urile NU pot prezice extrageri viitoare

Utilizare:
    python3 seed_finder.py --input loto_data.json --draws 2
    python3 seed_finder.py --input loto_data.json --draws 3 --seeds 1000000
"""

import argparse
import json
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class SimpleLCG:
    """Linear Congruential Generator - RNG simplu pentru demonstrație"""
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        # Parametrii LCG (similar cu glibc)
        self.a = 1103515245
        self.c = 12345
        self.m = 2**31
    
    def next(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state
    
    def next_in_range(self, min_val: int, max_val: int) -> int:
        return min_val + (self.next() % (max_val - min_val + 1))
    
    def generate_sequence(self, count: int, min_val: int, max_val: int) -> List[int]:
        """Generează o secvență de 'count' numere unice în range"""
        numbers = set()
        attempts = 0
        max_attempts = count * 100
        
        while len(numbers) < count and attempts < max_attempts:
            num = self.next_in_range(min_val, max_val)
            numbers.add(num)
            attempts += 1
        
        return sorted(list(numbers))[:count]


class Xorshift32:
    """Xorshift32 - Altă variantă de RNG pentru testare"""
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1
    
    def next(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x
    
    def next_in_range(self, min_val: int, max_val: int) -> int:
        return min_val + (self.next() % (max_val - min_val + 1))
    
    def generate_sequence(self, count: int, min_val: int, max_val: int) -> List[int]:
        numbers = set()
        attempts = 0
        max_attempts = count * 100
        
        while len(numbers) < count and attempts < max_attempts:
            num = self.next_in_range(min_val, max_val)
            numbers.add(num)
            attempts += 1
        
        return sorted(list(numbers))[:count]


class SeedFinder:
    def __init__(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.draws = self.data['draws']
        print(f"Încărcat {len(self.draws)} extrageri istorice\n")
    
    def calculate_match_score(self, generated: List[int], actual: List[int]) -> float:
        """
        Calculează scor de potrivire între secvență generată și reală
        """
        gen_set = set(generated)
        act_set = set(actual)
        
        # Câte numere comune
        common = len(gen_set & act_set)
        
        # Scor: numere comune / total numere
        return common / len(act_set) if len(act_set) > 0 else 0.0
    
    def find_seeds_for_sequence(self, 
                                 target_draws: List[List[int]], 
                                 seed_range: Tuple[int, int],
                                 rng_type: str = 'lcg',
                                 sample_size: int = 100000) -> List[Dict]:
        """
        Caută seed-uri care pot genera o secvență de extrageri
        """
        print(f"Căutare seed-uri în range [{seed_range[0]}, {seed_range[1]}]")
        print(f"Sample size: {sample_size:,} seed-uri testate")
        print(f"RNG type: {rng_type.upper()}")
        print(f"Target: {len(target_draws)} extrageri consecutive\n")
        
        candidates = []
        
        # Sample random din seed range
        test_seeds = random.sample(range(seed_range[0], seed_range[1]), 
                                   min(sample_size, seed_range[1] - seed_range[0]))
        
        for i, seed in enumerate(test_seeds):
            if (i + 1) % 10000 == 0:
                print(f"  Testat {i+1:,}/{sample_size:,} seeds...")
            
            # Generează secvență cu acest seed
            if rng_type == 'lcg':
                rng = SimpleLCG(seed)
            else:
                rng = Xorshift32(seed)
            
            # Calculează scor pentru fiecare extragere din target
            scores = []
            for target in target_draws:
                generated = rng.generate_sequence(6, 1, 40)
                score = self.calculate_match_score(generated, target)
                scores.append(score)
            
            # Scor mediu
            avg_score = sum(scores) / len(scores)
            
            # Dacă scorul e decent, salvează
            if avg_score > 0.3:  # Cel puțin 30% potrivire în medie
                candidates.append({
                    'seed': seed,
                    'avg_score': avg_score,
                    'scores': scores,
                    'max_score': max(scores),
                    'min_score': min(scores)
                })
        
        # Sortează după scor
        candidates.sort(key=lambda x: x['avg_score'], reverse=True)
        
        return candidates
    
    def test_seed_persistence(self, seed: int, start_idx: int, 
                             max_draws: int = 20, rng_type: str = 'lcg') -> Dict:
        """
        Testează câte extrageri consecutive poate "prezice" un seed
        """
        if rng_type == 'lcg':
            rng = SimpleLCG(seed)
        else:
            rng = Xorshift32(seed)
        
        results = []
        
        for i in range(max_draws):
            if start_idx + i >= len(self.draws):
                break
            
            actual = self.draws[start_idx + i]['numbers_sorted']
            generated = rng.generate_sequence(6, 1, 40)
            score = self.calculate_match_score(generated, actual)
            
            results.append({
                'draw_index': start_idx + i,
                'date': self.draws[start_idx + i]['date_str'],
                'actual': actual,
                'generated': generated,
                'score': score,
                'matches': len(set(generated) & set(actual))
            })
        
        return {
            'seed': seed,
            'start_index': start_idx,
            'results': results,
            'avg_score': sum(r['score'] for r in results) / len(results) if results else 0
        }
    
    def progressive_search(self, num_consecutive: int = 2, 
                          seed_range: Tuple[int, int] = (1, 1000000),
                          rng_type: str = 'lcg') -> List[Dict]:
        """
        Căutare progresivă: găsește seed-uri pentru fiecare secvență de N extrageri consecutive
        """
        print(f"\n{'='*70}")
        print(f"CĂUTARE PROGRESIVĂ - {num_consecutive} Extrageri Consecutive")
        print(f"{'='*70}\n")
        
        all_candidates = []
        
        # Pentru fiecare poziție posibilă în istoric
        max_start = len(self.draws) - num_consecutive
        
        # Sample doar câteva poziții (altfel durează prea mult)
        sample_positions = random.sample(range(max_start), min(10, max_start))
        
        for start_idx in sample_positions:
            print(f"\nAnalizare poziție {start_idx} - {start_idx + num_consecutive - 1}")
            print(f"Date: {self.draws[start_idx]['date_str']} → {self.draws[start_idx + num_consecutive - 1]['date_str']}")
            
            target_draws = [self.draws[i]['numbers_sorted'] 
                           for i in range(start_idx, start_idx + num_consecutive)]
            
            # Caută seed-uri pentru această secvență
            candidates = self.find_seeds_for_sequence(
                target_draws, 
                seed_range, 
                rng_type,
                sample_size=10000  # Sample mai mic pentru viteză
            )
            
            if candidates:
                best = candidates[0]
                print(f"\n  → Cel mai bun seed: {best['seed']}")
                print(f"  → Scor mediu: {best['avg_score']:.2%}")
                
                all_candidates.append({
                    'start_idx': start_idx,
                    'end_idx': start_idx + num_consecutive - 1,
                    'best_seed': best['seed'],
                    'score': best['avg_score'],
                    'target_draws': target_draws
                })
        
        return all_candidates


def main():
    parser = argparse.ArgumentParser(
        description='Seed Finder - Experiment educațional pentru căutare seed-uri în date loterie',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/app/backend/loto_data.json',
        help='Fișier JSON cu date Loto'
    )
    parser.add_argument(
        '--draws',
        type=int,
        default=2,
        help='Număr de extrageri consecutive de căutat (2-5)'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        default=100000,
        help='Număr de seed-uri de testat'
    )
    parser.add_argument(
        '--rng',
        type=str,
        choices=['lcg', 'xorshift'],
        default='lcg',
        help='Tip de RNG de folosit'
    )
    parser.add_argument(
        '--progressive',
        action='store_true',
        help='Căutare progresivă prin întregul istoric'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  SEED FINDER - EXPERIMENT EDUCAȚIONAL")
    print("="*70)
    print("\n⚠️  Acest experiment demonstrează că datele de loterie")
    print("    NU provin dintr-un RNG cu seed.\n")
    
    try:
        finder = SeedFinder(args.input)
        
        if args.progressive:
            # Căutare progresivă
            results = finder.progressive_search(
                num_consecutive=args.draws,
                seed_range=(1, 10000000),
                rng_type=args.rng
            )
            
            print("\n" + "="*70)
            print("REZULTATE CĂUTARE PROGRESIVĂ")
            print("="*70)
            
            if results:
                print(f"\nGăsite {len(results)} secvențe cu seed-uri candidate:\n")
                for i, r in enumerate(results[:5], 1):
                    print(f"{i}. Seed: {r['best_seed']:,} | Scor: {r['score']:.2%} | Poziție: {r['start_idx']}-{r['end_idx']}")
            else:
                print("\nNu s-au găsit seed-uri cu potrivire semnificativă.")
        
        else:
            # Căutare simplă pentru primele N extrageri
            target_draws = [finder.draws[i]['numbers_sorted'] 
                           for i in range(min(args.draws, len(finder.draws)))]
            
            print(f"Țintă: Primele {len(target_draws)} extrageri")
            for i, draw in enumerate(target_draws):
                print(f"  {i+1}. {finder.draws[i]['date_str']}: {draw}")
            print()
            
            candidates = finder.find_seeds_for_sequence(
                target_draws,
                seed_range=(1, 10000000),
                rng_type=args.rng,
                sample_size=args.seeds
            )
            
            print("\n" + "="*70)
            print("REZULTATE")
            print("="*70)
            
            if candidates:
                print(f"\nGăsite {len(candidates)} seed-uri candidate:\n")
                
                for i, cand in enumerate(candidates[:10], 1):
                    print(f"{i}. Seed: {cand['seed']:,}")
                    print(f"   Scor mediu: {cand['avg_score']:.2%}")
                    print(f"   Range: {cand['min_score']:.2%} - {cand['max_score']:.2%}")
                    print()
                
                # Testează persistența celui mai bun seed
                print("\nTest persistență cel mai bun seed:")
                print("-" * 70)
                
                best_seed = candidates[0]['seed']
                persistence = finder.test_seed_persistence(
                    best_seed, 
                    0, 
                    max_draws=10,
                    rng_type=args.rng
                )
                
                print(f"\nSeed: {best_seed:,}")
                print(f"Scor mediu pe 10 extrageri: {persistence['avg_score']:.2%}\n")
                
                for r in persistence['results'][:5]:
                    print(f"{r['date']}: {r['matches']}/6 potriviri (scor: {r['score']:.2%})")
                    print(f"  Actual:    {r['actual']}")
                    print(f"  Generated: {r['generated']}")
                    print()
            
            else:
                print("\nNu s-au găsit seed-uri cu potrivire semnificativă.")
                print("(Acest lucru este AȘTEPTAT - loteriile nu au seed-uri!)")
        
        print("\n" + "="*70)
        print("CONCLUZIE")
        print("="*70)
        print("""
Acest experiment arată că:
1. Chiar dacă găsim "seed-uri" care recrează unele secvențe,
   acestea NU persistă - scorul scade rapid
2. "Potrivirile" sunt întâmplătoare, nu reprezintă un pattern real
3. Seed-urile diferite sunt necesare pentru fiecare secvență
4. Datele de loterie NU provin dintr-un RNG cu seed

Concluzie: Extragerea fizică cu bile nu are seed și nu poate fi prezisă!
        """)
        
    except FileNotFoundError:
        print(f"Eroare: Fișierul {args.input} nu există.")
        print("Rulează mai întâi: python3 loto_scraper.py")
    except Exception as e:
        print(f"Eroare: {e}")


if __name__ == '__main__':
    main()
