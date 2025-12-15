#!/usr/bin/env python3
"""
Analizor statistic pentru date Loto 5/40

Utilizare:
    python3 loto_analyzer.py --input loto_data.json
    python3 loto_analyzer.py --input loto_data.json --top 15
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple
import statistics


class LotoAnalyzer:
    def __init__(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.draws = self.data['draws']
        print(f"Încărcat {len(self.draws)} extrageri")
    
    def analyze_frequency(self, top_n: int = 10) -> Dict:
        """
        Analizează frecvența numerelor
        """
        all_numbers = []
        for draw in self.draws:
            all_numbers.extend(draw['numbers'])
        
        counter = Counter(all_numbers)
        most_common = counter.most_common(top_n)
        least_common = counter.most_common()[-top_n:]
        
        return {
            'most_common': most_common,
            'least_common': least_common,
            'all_frequencies': dict(counter)
        }
    
    def analyze_pairs(self, top_n: int = 10) -> List[Tuple]:
        """
        Analizează perechile de numere care apar împreună
        """
        pairs = Counter()
        
        for draw in self.draws:
            nums = sorted(draw['numbers'])
            # Generează toate perechile
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    pair = (nums[i], nums[j])
                    pairs[pair] += 1
        
        return pairs.most_common(top_n)
    
    def analyze_triplets(self, top_n: int = 10) -> List[Tuple]:
        """
        Analizează tripletele de numere
        """
        triplets = Counter()
        
        for draw in self.draws:
            nums = sorted(draw['numbers'])
            # Generează toate tripletele
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    for k in range(j + 1, len(nums)):
                        triplet = (nums[i], nums[j], nums[k])
                        triplets[triplet] += 1
        
        return triplets.most_common(top_n)
    
    def analyze_gaps(self) -> Dict:
        """
        Analizează intervalele între apariții pentru fiecare număr
        """
        last_seen = {}  # număr -> indexul ultimei apariții
        gaps = defaultdict(list)  # număr -> lista de intervale
        
        for idx, draw in enumerate(self.draws):
            for num in draw['numbers']:
                if num in last_seen:
                    gap = idx - last_seen[num]
                    gaps[num].append(gap)
                last_seen[num] = idx
        
        # Calculează statistici pentru fiecare număr
        gap_stats = {}
        for num in range(1, 41):
            if num in gaps and gaps[num]:
                gap_list = gaps[num]
                gap_stats[num] = {
                    'avg_gap': statistics.mean(gap_list),
                    'min_gap': min(gap_list),
                    'max_gap': max(gap_list),
                    'median_gap': statistics.median(gap_list)
                }
        
        return gap_stats
    
    def analyze_hot_cold(self, recent_draws: int = 50) -> Dict:
        """
        Identifică numere 'fierbinți' (frecvente recent) și 'reci' (rare recent)
        """
        # Numere din ultimele N extrageri
        recent = self.draws[-recent_draws:] if len(self.draws) >= recent_draws else self.draws
        
        recent_numbers = []
        for draw in recent:
            recent_numbers.extend(draw['numbers'])
        
        recent_counter = Counter(recent_numbers)
        
        # Top 10 hot și cold
        hot = recent_counter.most_common(10)
        all_recent = recent_counter.most_common()
        cold = all_recent[-10:] if len(all_recent) >= 10 else all_recent
        
        return {
            'hot_numbers': hot,
            'cold_numbers': cold,
            'period': f"ultimele {len(recent)} extrageri"
        }
    
    def analyze_patterns(self) -> Dict:
        """
        Analizează pattern-uri (par/impar, mic/mare, etc.)
        """
        even_odd = []
        low_high = []  # 1-20 vs 21-40
        
        for draw in self.draws:
            nums = draw['numbers']
            even_count = sum(1 for n in nums if n % 2 == 0)
            odd_count = len(nums) - even_count
            even_odd.append((even_count, odd_count))
            
            low_count = sum(1 for n in nums if n <= 20)
            high_count = len(nums) - low_count
            low_high.append((low_count, high_count))
        
        # Distribuția par/impar
        even_odd_dist = Counter(even_odd)
        low_high_dist = Counter(low_high)
        
        return {
            'even_odd_distribution': dict(even_odd_dist),
            'low_high_distribution': dict(low_high_dist)
        }
    
    def print_analysis(self, top_n: int = 10):
        """
        Afișează analiza completă
        """
        print("\n" + "="*70)
        print("ANALIZĂ STATISTICĂ LOTO 5/40")
        print("="*70)
        print(f"Total extrageri analizate: {len(self.draws)}")
        if self.draws:
            print(f"Perioadă: {self.draws[0]['date_str']} → {self.draws[-1]['date_str']}")
        
        # 1. Frecvența numerelor
        print("\n" + "-"*70)
        print(f"1. TOP {top_n} NUMERE CELE MAI FRECVENTE")
        print("-"*70)
        freq = self.analyze_frequency(top_n)
        for num, count in freq['most_common']:
            percentage = (count / (len(self.draws) * 6)) * 100
            print(f"  {num:2d}: {count:4d} apariții ({percentage:.2f}%)")
        
        # 2. Numere rare
        print(f"\n{top_n}. NUMERE CELE MAI RARE")
        for num, count in freq['least_common']:
            percentage = (count / (len(self.draws) * 6)) * 100
            print(f"  {num:2d}: {count:4d} apariții ({percentage:.2f}%)")
        
        # 3. Perechi frecvente
        print("\n" + "-"*70)
        print(f"2. TOP {top_n} PERECHI FRECVENTE")
        print("-"*70)
        pairs = self.analyze_pairs(top_n)
        for pair, count in pairs:
            print(f"  {pair[0]:2d}-{pair[1]:2d}: {count:3d} apariții împreună")
        
        # 4. Triplete
        print("\n" + "-"*70)
        print(f"3. TOP {min(5, top_n)} TRIPLETE FRECVENTE")
        print("-"*70)
        triplets = self.analyze_triplets(min(5, top_n))
        for triplet, count in triplets:
            print(f"  {triplet[0]:2d}-{triplet[1]:2d}-{triplet[2]:2d}: {count:3d} apariții împreună")
        
        # 5. Numere fierbinți și reci
        print("\n" + "-"*70)
        print("4. NUMERE FIERBINȚI vs RECI (ultimele 50 extrageri)")
        print("-"*70)
        hot_cold = self.analyze_hot_cold(50)
        
        print(f"\nNumere FIERBINȚI ({hot_cold['period']}):")
        for num, count in hot_cold['hot_numbers']:
            print(f"  {num:2d}: {count:3d} apariții")
        
        print(f"\nNumere RECI ({hot_cold['period']}):")
        for num, count in hot_cold['cold_numbers']:
            print(f"  {num:2d}: {count:3d} apariții")
        
        # 6. Pattern-uri
        print("\n" + "-"*70)
        print("5. PATTERN-URI (PAR/IMPAR, MIC/MARE)")
        print("-"*70)
        patterns = self.analyze_patterns()
        
        print("\nDistribuție PAR/IMPAR (format: par, impar):")
        for pattern, count in sorted(patterns['even_odd_distribution'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / len(self.draws)) * 100
            print(f"  {pattern[0]} par, {pattern[1]} impar: {count:3d} extrageri ({percentage:.1f}%)")
        
        print("\nDistribuție MIC/MARE (1-20 vs 21-40):")
        for pattern, count in sorted(patterns['low_high_distribution'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / len(self.draws)) * 100
            print(f"  {pattern[0]} mici, {pattern[1]} mari: {count:3d} extrageri ({percentage:.1f}%)")
        
        # 7. Disclaimer
        print("\n" + "="*70)
        print("IMPORTANT - DISCLAIMER")
        print("="*70)
        print("""
Aceste statistici reflectă doar datele istorice și NU pot prezice 
viitoarele extrageri. Loto 5/40 folosește extragere FIZICĂ cu bile,
fiecare extragere fiind independentă și complet aleatoare.

Numere "fierbinți" sau "reci" sunt doar artefacte statistice și nu 
au nicio putere predictivă.
        """)


def main():
    parser = argparse.ArgumentParser(
        description='Analizează statistic datele Loto 5/40'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/app/backend/loto_data.json',
        help='Fișier JSON cu date Loto'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Număr de top rezultate de afișat'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = LotoAnalyzer(args.input)
        analyzer.print_analysis(args.top)
    except FileNotFoundError:
        print(f"Eroare: Fișierul {args.input} nu există.")
        print("Rulează mai întâi: python3 loto_scraper.py")
    except Exception as e:
        print(f"Eroare: {e}")


if __name__ == '__main__':
    main()
