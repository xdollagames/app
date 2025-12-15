#!/usr/bin/env python3
"""
Seed Evaluator - Evalu

ează "calitatea" seed-urilor găsite

Testează:
1. Persistența - câte extrageri consecutive poate "prezice"
2. Consistența - dacă același seed funcționează în părți diferite din istoric
3. Calitatea match-urilor - cât de bune sunt potrivirile

Utilizare:
    python3 seed_evaluator.py --seeds 12345,67890,111213
    python3 seed_evaluator.py --auto-find --top 5
"""

import argparse
import json
from typing import List, Dict
from seed_finder import SimpleLCG, Xorshift32, SeedFinder
import statistics


class SeedEvaluator:
    def __init__(self, data_file: str):
        self.finder = SeedFinder(data_file)
        self.draws = self.finder.draws
    
    def evaluate_seed_quality(self, seed: int, rng_type: str = 'lcg') -> Dict:
        """
        Evaluează comprehensive un seed
        """
        print(f"\nEvaluare seed: {seed:,}")
        print("=" * 60)
        
        if rng_type == 'lcg':
            rng = SimpleLCG(seed)
        else:
            rng = Xorshift32(seed)
        
        # Test 1: Persistență continuă
        print("\n1. Test Persistență (primele 20 extrageri):")
        scores = []
        matches_count = []
        
        for i in range(min(20, len(self.draws))):
            actual = self.draws[i]['numbers_sorted']
            generated = rng.generate_sequence(6, 1, 40)
            
            common = len(set(generated) & set(actual))
            score = common / 6.0
            
            scores.append(score)
            matches_count.append(common)
        
        avg_score = statistics.mean(scores)
        max_matches = max(matches_count)
        avg_matches = statistics.mean(matches_count)
        
        print(f"   Scor mediu: {avg_score:.2%}")
        print(f"   Match-uri medii: {avg_matches:.1f}/6")
        print(f"   Max match-uri: {max_matches}/6")
        print(f"   Persistență: {self._calculate_persistence(scores)} extrageri")
        
        # Test 2: Consistență în diferite părți ale istoricului
        print("\n2. Test Consistență (sample din istoric):")
        consistency_scores = []
        
        # Testează în 5 puncte diferite din istoric
        test_points = [0, len(self.draws)//4, len(self.draws)//2, 
                       3*len(self.draws)//4, len(self.draws)-10]
        
        for start in test_points:
            if start >= len(self.draws):
                continue
            
            # Re-inițializează RNG cu același seed
            if rng_type == 'lcg':
                test_rng = SimpleLCG(seed)
            else:
                test_rng = Xorshift32(seed)
            
            # Generează până la acel punct
            for _ in range(start):
                test_rng.generate_sequence(6, 1, 40)
            
            # Testează următoarele 5 extrageri
            test_scores = []
            for i in range(min(5, len(self.draws) - start)):
                actual = self.draws[start + i]['numbers_sorted']
                generated = test_rng.generate_sequence(6, 1, 40)
                score = len(set(generated) & set(actual)) / 6.0
                test_scores.append(score)
            
            if test_scores:
                consistency_scores.append(statistics.mean(test_scores))
        
        consistency = statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0
        print(f"   Consistență (stdev): {consistency:.3f} (mai mic = mai consistent)")
        print(f"   Scoruri în puncte diferite: {[f'{s:.2%}' for s in consistency_scores]}")
        
        # Test 3: Distribuție match-uri
        print("\n3. Distribuție Match-uri:")
        match_dist = {i: matches_count.count(i) for i in range(7)}
        for matches, count in sorted(match_dist.items()):
            bar = '█' * (count * 2)
            print(f"   {matches}/6: {bar} ({count})")
        
        # Scor final compozit
        quality_score = (
            avg_score * 0.5 +  # 50% - scor mediu
            (1 - consistency) * 0.3 +  # 30% - consistență
            (max_matches / 6) * 0.2  # 20% - cel mai bun match
        )
        
        return {
            'seed': seed,
            'avg_score': avg_score,
            'avg_matches': avg_matches,
            'max_matches': max_matches,
            'consistency': consistency,
            'persistence': self._calculate_persistence(scores),
            'quality_score': quality_score,
            'match_distribution': match_dist
        }
    
    def _calculate_persistence(self, scores: List[float], threshold: float = 0.3) -> int:
        """
        Calculează câte extrageri consecutive au scor > threshold
        """
        count = 0
        for score in scores:
            if score >= threshold:
                count += 1
            else:
                break
        return count
    
    def compare_seeds(self, seeds: List[int], rng_type: str = 'lcg') -> None:
        """
        Compară mai mulți seed
        """
        print("\n" + "="*70)
        print("COMPARAȚIE SEED-URI")
        print("="*70)
        
        results = []
        for seed in seeds:
            result = self.evaluate_seed_quality(seed, rng_type)
            results.append(result)
        
        # Sortează după quality score
        results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        print("\n" + "="*70)
        print("CLASAMENT FINAL")
        print("="*70)
        print()
        print(f"{'Rank':<6} {'Seed':<12} {'Quality':<10} {'Avg Match':<12} {'Persist.':<10}")
        print("-" * 70)
        
        for i, r in enumerate(results, 1):
            print(f"{i:<6} {r['seed']:<12,} {r['quality_score']:<10.2%} "
                  f"{r['avg_matches']:<12.1f}/6 {r['persistence']:<10}")
        
        print("\n" + "="*70)
        print("OBSERVAȚII")
        print("="*70)
        print("""
Dacă observi:
• Scoruri mici (<20%) - Normal pentru date aleatorii
• Persistență scăzută (1-3 extrageri) - Confirmare că nu e RNG
• Inconsistență ridicată - Seed-urile nu "funcționează" consistent
• Distribuție uniformă a match-urilor - Caracteristic aleatoriu pur

Concluziile arată că datele NU provin dintr-un RNG cu seed!
        """)
    
    def find_and_evaluate_best(self, count: int = 5, rng_type: str = 'lcg') -> None:
        """
        Găsește automat cei mai buni seed și îi evaluează
        """
        print("\nCăutare automată seed-uri candidate...")
        
        # Caută seed-uri pentru primele 3 extrageri
        target_draws = [self.draws[i]['numbers_sorted'] for i in range(3)]
        
        candidates = self.finder.find_seeds_for_sequence(
            target_draws,
            seed_range=(1, 5000000),
            rng_type=rng_type,
            sample_size=50000
        )
        
        if not candidates:
            print("\nNu s-au găsit seed-uri candidate.")
            return
        
        # Ia top N
        top_seeds = [c['seed'] for c in candidates[:count]]
        
        print(f"\nGăsite {len(candidates)} candidați. Evaluare top {count}...")
        
        self.compare_seeds(top_seeds, rng_type)


def main():
    parser = argparse.ArgumentParser(
        description='Seed Evaluator - Evaluează calitatea seed-urilor găsite'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/app/backend/loto_data.json',
        help='Fișier JSON cu date Loto'
    )
    parser.add_argument(
        '--seeds',
        type=str,
        help='Lista de seed-uri separate prin virgulă (ex: 12345,67890)'
    )
    parser.add_argument(
        '--auto-find',
        action='store_true',
        help='Găsește automat cei mai buni seed'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=5,
        help='Număr de seed-uri de evaluat (cu --auto-find)'
    )
    parser.add_argument(
        '--rng',
        type=str,
        choices=['lcg', 'xorshift'],
        default='lcg',
        help='Tip de RNG'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  SEED EVALUATOR - Analiză Calitate Seed-uri")
    print("="*70)
    
    try:
        evaluator = SeedEvaluator(args.input)
        
        if args.auto_find:
            evaluator.find_and_evaluate_best(args.top, args.rng)
        elif args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
            evaluator.compare_seeds(seeds, args.rng)
        else:
            print("\nEroare: Specifică --seeds sau --auto-find")
            print("\nExemple:")
            print("  python3 seed_evaluator.py --seeds 12345,67890")
            print("  python3 seed_evaluator.py --auto-find --top 5")
    
    except FileNotFoundError:
        print(f"Eroare: Fișierul {args.input} nu există.")
        print("Rulează mai întâi: python3 loto_scraper.py")
    except Exception as e:
        print(f"Eroare: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
