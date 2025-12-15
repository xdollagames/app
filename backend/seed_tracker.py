#!/usr/bin/env python3
"""
Seed Tracker - Urmărește "cei mai buni" seed-uri de-a lungul timpului

Analizează:
1. Care seed are cele mai multe "hit-uri" (match-uri bune) în istoric
2. Dacă seed-urile "persistente" există
3. Evoluția performanței seed-urilor în timp

Utilizare:
    python3 seed_tracker.py --track 100000
    python3 seed_tracker.py --compare-evolution
"""

import argparse
import json
from typing import List, Dict, Tuple
from seed_finder import SimpleLCG, Xorshift32, SeedFinder
from collections import defaultdict
import random


class SeedTracker:
    def __init__(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.draws = self.data['draws']
        print(f"Încărcat {len(self.draws)} extrageri istorice\n")
    
    def track_seed_performance(self, seed: int, rng_type: str = 'lcg') -> Dict:
        """
        Urmărește performanța unui seed pe întreg istoricul
        """
        if rng_type == 'lcg':
            rng = SimpleLCG(seed)
        else:
            rng = Xorshift32(seed)
        
        results = []
        total_matches = 0
        hits = []  # Extrageri cu match-uri bune (>=3)
        
        for i, draw in enumerate(self.draws):
            actual = draw['numbers_sorted']
            generated = rng.generate_sequence(6, 1, 40)
            matches = len(set(generated) & set(actual))
            
            total_matches += matches
            
            if matches >= 3:
                hits.append({
                    'index': i,
                    'date': draw['date_str'],
                    'matches': matches,
                    'actual': actual,
                    'generated': generated
                })
            
            results.append(matches)
        
        return {
            'seed': seed,
            'total_draws': len(self.draws),
            'total_matches': total_matches,
            'avg_matches': total_matches / len(self.draws),
            'hits': hits,
            'hit_rate': len(hits) / len(self.draws),
            'results': results
        }
    
    def find_best_persistent_seeds(self, num_seeds: int = 100000, 
                                    rng_type: str = 'lcg') -> List[Dict]:
        """
        Găsește seed-urile cu cele mai multe "hit-uri" în întreg istoricul
        """
        print(f"Testare {num_seeds:,} seed-uri random...")
        print(f"Criteriu: Număr maxim de extrageri cu >=3 match-uri\n")
        
        test_seeds = random.sample(range(1, 10000000), num_seeds)
        candidates = []
        
        for i, seed in enumerate(test_seeds):
            if (i + 1) % 10000 == 0:
                print(f"  Progress: {i+1:,}/{num_seeds:,}")
            
            perf = self.track_seed_performance(seed, rng_type)
            
            # Doar dacă are hit-uri decente
            if len(perf['hits']) > 0:
                candidates.append({
                    'seed': seed,
                    'hit_count': len(perf['hits']),
                    'hit_rate': perf['hit_rate'],
                    'avg_matches': perf['avg_matches'],
                    'best_matches': max(h['matches'] for h in perf['hits']) if perf['hits'] else 0
                })
        
        # Sortează după număr de hit-uri
        candidates.sort(key=lambda x: (x['hit_count'], x['best_matches']), reverse=True)
        
        return candidates
    
    def analyze_seed_evolution(self, seed: int, window_size: int = 10, 
                              rng_type: str = 'lcg') -> Dict:
        """
        Analizează cum evoluează performanța unui seed în timp
        """
        perf = self.track_seed_performance(seed, rng_type)
        results = perf['results']
        
        # Calculează medie mobilă
        moving_avg = []
        for i in range(len(results) - window_size + 1):
            window = results[i:i+window_size]
            moving_avg.append(sum(window) / len(window))
        
        # Detectă trend
        if len(moving_avg) > 1:
            trend = (moving_avg[-1] - moving_avg[0]) / len(moving_avg)
        else:
            trend = 0
        
        return {
            'seed': seed,
            'moving_avg': moving_avg,
            'trend': trend,
            'trend_direction': 'crescator' if trend > 0 else 'descrescator' if trend < 0 else 'stabil'
        }
    
    def compare_multiple_evolutions(self, seeds: List[int], rng_type: str = 'lcg') -> None:
        """
        Compară evoluția mai multor seed-uri
        """
        print("\n" + "="*70)
        print("ANALIZĂ EVOLUȚIE SEED-URI")
        print("="*70)
        
        for seed in seeds:
            print(f"\nSeed: {seed:,}")
            print("-" * 60)
            
            perf = self.track_seed_performance(seed, rng_type)
            evol = self.analyze_seed_evolution(seed, 10, rng_type)
            
            print(f"Match-uri medii: {perf['avg_matches']:.2f}/6")
            print(f"Hit-uri (>=3 match): {len(perf['hits'])} din {perf['total_draws']} ({perf['hit_rate']:.1%})")
            print(f"Trend: {evol['trend_direction']} ({evol['trend']:.4f})")
            
            if perf['hits']:
                print(f"\nTop 3 hit-uri:")
                for hit in perf['hits'][:3]:
                    print(f"  {hit['date']}: {hit['matches']}/6 match-uri")
                    print(f"    Actual:    {hit['actual']}")
                    print(f"    Generated: {hit['generated']}")
            
            # Mini-grafic ASCII pentru evoluție
            if len(evol['moving_avg']) > 0:
                print(f"\nEvoluție (medie mobilă 10 extrageri):")
                self._print_ascii_chart(evol['moving_avg'][:50])  # Primele 50 puncte
    
    def _print_ascii_chart(self, data: List[float], width: int = 60, height: int = 10) -> None:
        """
        Afișează un grafic ASCII simplu
        """
        if not data:
            return
        
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val > min_val else 1
        
        # Normalize la height
        normalized = [(d - min_val) / range_val * height for d in data]
        
        # Print grafic
        for h in range(height, -1, -1):
            line = ""
            for n in normalized:
                if n >= h:
                    line += "█"
                else:
                    line += " "
            
            val = min_val + (h / height) * range_val
            print(f"{val:5.2f} | {line}")
        
        print(f"{'      '}"+" " + "-" * len(normalized))
        print(f"      Start{' ' * (len(normalized) - 10)}End")


def main():
    parser = argparse.ArgumentParser(
        description='Seed Tracker - Urmărește performanța seed-urilor în timp'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/app/backend/loto_data.json',
        help='Fișier JSON cu date Loto'
    )
    parser.add_argument(
        '--track',
        type=int,
        metavar='N',
        help='Testează N seed-uri random și găsește cei mai buni'
    )
    parser.add_argument(
        '--compare-evolution',
        action='store_true',
        help='Compară evoluția seed-urilor populate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Analizează un seed specific'
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
    print("  SEED TRACKER - Urmărire Performanță Seed-uri")
    print("="*70)
    print("\n⚠️  Experiment educațional - datele de loterie NU au seed-uri!\n")
    
    try:
        tracker = SeedTracker(args.input)
        
        if args.track:
            # Găsește cei mai buni seed-uri
            candidates = tracker.find_best_persistent_seeds(args.track, args.rng)
            
            print("\n" + "="*70)
            print("CEI MAI BUNI SEED-URI GĂSIȚI")
            print("="*70)
            
            if candidates:
                print(f"\nTop 10 seed-uri cu cele mai multe hit-uri (>=3 match):\n")
                print(f"{'Rank':<6} {'Seed':<12} {'Hits':<8} {'Hit Rate':<12} {'Avg Match':<12} {'Best'}")
                print("-" * 70)
                
                for i, c in enumerate(candidates[:10], 1):
                    print(f"{i:<6} {c['seed']:<12,} {c['hit_count']:<8} "
                          f"{c['hit_rate']:<12.2%} {c['avg_matches']:<12.2f} {c['best_matches']}/6")
                
                # Analizează top 3
                print("\n" + "="*70)
                print("ANALIZĂ DETALIATĂ TOP 3")
                print("="*70)
                
                top_seeds = [c['seed'] for c in candidates[:3]]
                tracker.compare_multiple_evolutions(top_seeds, args.rng)
                
            else:
                print("\nNu s-au găsit seed-uri cu hit-uri semnificative.")
        
        elif args.compare_evolution:
            # Compară evoluția unor seed-uri sample
            sample_seeds = random.sample(range(1, 10000000), 5)
            tracker.compare_multiple_evolutions(sample_seeds, args.rng)
        
        elif args.seed:
            # Analiză un seed specific
            print(f"\nAnalizare seed: {args.seed:,}")
            print("=" * 60)
            
            perf = tracker.track_seed_performance(args.seed, args.rng)
            evol = tracker.analyze_seed_evolution(args.seed, 10, args.rng)
            
            print(f"\nStatistici generale:")
            print(f"  Total match-uri: {perf['total_matches']} din {perf['total_draws'] * 6} posibile")
            print(f"  Medie match-uri: {perf['avg_matches']:.2f}/6 per extragere")
            print(f"  Hit-uri (>=3): {len(perf['hits'])} ({perf['hit_rate']:.1%})")
            
            if perf['hits']:
                print(f"\nToate hit-urile (>=3 match-uri):")
                for hit in perf['hits']:
                    print(f"  {hit['date']}: {hit['matches']}/6")
                    print(f"    {hit['actual']} vs {hit['generated']}")
            
            print(f"\nEvoluție:")
            print(f"  Trend: {evol['trend_direction']}")
            tracker._print_ascii_chart(evol['moving_avg'])
        
        else:
            print("\nEroare: Specifică --track, --compare-evolution, sau --seed")
            print("\nExemple:")
            print("  python3 seed_tracker.py --track 100000")
            print("  python3 seed_tracker.py --seed 12345")
            print("  python3 seed_tracker.py --compare-evolution")
        
        print("\n" + "="*70)
        print("INTERPRETĂRI")
        print("="*70)
        print("""
Ce ar trebui să observi:

1. Hit-uri rare: Chiar și "cei mai buni" seed au puține hit-uri
   → Confirmare că nu există un seed real

2. Trend instabil: Performanța nu este consistentă în timp
   → Nu există persistă - caracteristic datelor aleatorii

3. Match-uri medii scăzute (~1-2/6): Aproximativ șansa randomă
   → Șansa matematică de 1 match random = 1/6 * 6 = 1

4. Seed-uri diferite sunt "cei mai buni" pentru perioade diferite
   → Nu există un "seed universal"

Concluzie: Datele NU provin dintr-un RNG - sunt extrageri fizice aleatorii!
        """)
    
    except FileNotFoundError:
        print(f"Eroare: Fișierul {args.input} nu există.")
        print("Rulează mai întâi: python3 loto_scraper.py")
    except Exception as e:
        print(f"Eroare: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
