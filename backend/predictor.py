#!/usr/bin/env python3
"""
Generator "inteligent" de combinații Loto 5/40

Genereaza combinații bazate pe diverse strategii statistice.
⚠️ IMPORTANT: Aceste strategii NU pot prezice rezultatele.
Fiecare extragere este independentă și aleatoare.

Utilizare:
    python3 predictor.py --strategy frequency
    python3 predictor.py --strategy balanced --count 5
    python3 predictor.py --strategy all
"""

import argparse
import json
import random
from collections import Counter
from typing import List, Set


class LottoPredictor:
    def __init__(self, data_file: str = None):
        self.numbers_range = range(1, 41)  # Loto 5/40
        self.pick_count = 6
        
        if data_file:
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                self.draws = self.data['draws']
                print(f"Încărcat {len(self.draws)} extrageri istorice\n")
            except FileNotFoundError:
                print(f"⚠ Fișierul {data_file} nu există. Folosim doar random.\n")
                self.draws = []
        else:
            self.draws = []
    
    def strategy_frequency(self, top_percent: float = 0.4) -> Set[int]:
        """
        Strategie: Alege din numerele cele mai frecvente istoric
        """
        if not self.draws:
            return self._random_selection()
        
        all_numbers = []
        for draw in self.draws:
            all_numbers.extend(draw['numbers'])
        
        counter = Counter(all_numbers)
        
        # Ia top X% numere
        top_count = int(len(counter) * top_percent)
        most_common = counter.most_common(top_count)
        top_numbers = [num for num, _ in most_common]
        
        # Alege random din top
        return set(random.sample(top_numbers, self.pick_count))
    
    def strategy_balanced(self) -> Set[int]:
        """
        Strategie: Echilibrează par/impar și mic/mare (1-20 vs 21-40)
        """
        numbers = set()
        
        # 3 pare, 3 impare SAU 4-2 sau 2-4
        even_count = random.choice([2, 3, 4])
        odd_count = self.pick_count - even_count
        
        evens = [n for n in self.numbers_range if n % 2 == 0]
        odds = [n for n in self.numbers_range if n % 2 != 0]
        
        selected_evens = random.sample(evens, even_count)
        selected_odds = random.sample(odds, odd_count)
        
        numbers.update(selected_evens)
        numbers.update(selected_odds)
        
        # Asigură-te că avem și din 1-20 și din 21-40
        low = [n for n in numbers if n <= 20]
        high = [n for n in numbers if n > 20]
        
        # Dacă e dezechilibrat, ajustează
        if len(low) == 0 or len(high) == 0:
            return self._random_selection()
        
        return numbers
    
    def strategy_hot_numbers(self, recent_draws: int = 50) -> Set[int]:
        """
        Strategie: Alege din numerele "fierbinți" (frecvente recent)
        """
        if not self.draws:
            return self._random_selection()
        
        recent = self.draws[-recent_draws:] if len(self.draws) >= recent_draws else self.draws
        
        recent_numbers = []
        for draw in recent:
            recent_numbers.extend(draw['numbers'])
        
        counter = Counter(recent_numbers)
        
        # Top 15 cele mai frecvente
        hot = counter.most_common(15)
        hot_numbers = [num for num, _ in hot]
        
        if len(hot_numbers) >= self.pick_count:
            return set(random.sample(hot_numbers, self.pick_count))
        else:
            return self._random_selection()
    
    def strategy_cold_numbers(self, recent_draws: int = 50) -> Set[int]:
        """
        Strategie: Alege din numerele "reci" (rare recent)
        """
        if not self.draws:
            return self._random_selection()
        
        recent = self.draws[-recent_draws:] if len(self.draws) >= recent_draws else self.draws
        
        recent_numbers = []
        for draw in recent:
            recent_numbers.extend(draw['numbers'])
        
        counter = Counter(recent_numbers)
        
        # Toate numerele ordonate de la cele mai rare
        all_counts = counter.most_common()
        all_counts.reverse()  # De la cele mai rare
        
        cold_numbers = [num for num, _ in all_counts[:15]]
        
        if len(cold_numbers) >= self.pick_count:
            return set(random.sample(cold_numbers, self.pick_count))
        else:
            return self._random_selection()
    
    def strategy_mixed(self) -> Set[int]:
        """
        Strategie: Combină numere fierbinți, reci și random
        """
        if not self.draws:
            return self._random_selection()
        
        numbers = set()
        
        # 2 fierbinți
        hot = list(self.strategy_hot_numbers(50))
        if len(hot) >= 2:
            numbers.update(random.sample(hot, 2))
        
        # 2 reci
        cold = list(self.strategy_cold_numbers(50))
        cold = [n for n in cold if n not in numbers]
        if len(cold) >= 2:
            numbers.update(random.sample(cold, 2))
        
        # Completează cu random
        remaining = [n for n in self.numbers_range if n not in numbers]
        needed = self.pick_count - len(numbers)
        
        if needed > 0 and len(remaining) >= needed:
            numbers.update(random.sample(remaining, needed))
        
        return numbers
    
    def strategy_avoid_recent(self, avoid_draws: int = 3) -> Set[int]:
        """
        Strategie: Evită numerele din ultimele N extrageri
        (Bazat pe "gamblers fallacy" - nu are bază matematică!)
        """
        if not self.draws:
            return self._random_selection()
        
        recent = self.draws[-avoid_draws:] if len(self.draws) >= avoid_draws else []
        
        recent_numbers = set()
        for draw in recent:
            recent_numbers.update(draw['numbers'])
        
        available = [n for n in self.numbers_range if n not in recent_numbers]
        
        if len(available) >= self.pick_count:
            return set(random.sample(available, self.pick_count))
        else:
            return self._random_selection()
    
    def _random_selection(self) -> Set[int]:
        """
        Selecție complet aleatoare
        """
        return set(random.sample(list(self.numbers_range), self.pick_count))
    
    def generate(self, strategy: str, count: int = 1) -> List[List[int]]:
        """
        Generează combinații folosind strategia specificată
        """
        strategies = {
            'frequency': self.strategy_frequency,
            'balanced': self.strategy_balanced,
            'hot': self.strategy_hot_numbers,
            'cold': self.strategy_cold_numbers,
            'mixed': self.strategy_mixed,
            'avoid_recent': self.strategy_avoid_recent,
            'random': self._random_selection
        }
        
        if strategy not in strategies:
            print(f"Strategie necunoscută: {strategy}")
            return []
        
        results = []
        for _ in range(count):
            combination = strategies[strategy]()
            results.append(sorted(list(combination)))
        
        return results


def print_combinations(combinations: List[List[int]], strategy: str):
    """
    Afișează combinațiile generate
    """
    print(f"\nCombinatii generate cu strategia '{strategy}':")
    print("-" * 50)
    
    for i, combo in enumerate(combinations, 1):
        combo_str = " - ".join([f"{n:2d}" for n in combo])
        print(f"  {i}. {combo_str}")


def main():
    parser = argparse.ArgumentParser(
        description='Generator combinații Loto 5/40',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategii disponibile:
  frequency    - Numere frecvente istoric
  balanced     - Echilibru par/impar, mic/mare
  hot          - Numere fierbinți (frecvente recent)
  cold         - Numere reci (rare recent)
  mixed        - Combinație de strategii
  avoid_recent - Evită numerele recente
  random       - Selecție aleatoare
  all          - Generează câte o combinație din fiecare strategie

⚠️  DISCLAIMER: Aceste strategii sunt doar pentru distracție/experiment.
    Fiecare extragere Loto este independentă și complet aleatoare.
    Nu există o strategie care să crească șansele de câștig!
        """
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='random',
        help='Strategia de generare'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=1,
        help='Număr de combinații de generat'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='/app/backend/loto_data.json',
        help='Fișier JSON cu date istorice'
    )
    
    args = parser.parse_args()
    
    predictor = LottoPredictor(args.data)
    
    print("="*70)
    print("GENERATOR COMBINAȚII LOTO 5/40")
    print("="*70)
    
    if args.strategy == 'all':
        strategies = ['frequency', 'balanced', 'hot', 'cold', 'mixed', 'avoid_recent', 'random']
        for strat in strategies:
            combinations = predictor.generate(strat, 1)
            print_combinations(combinations, strat)
    else:
        combinations = predictor.generate(args.strategy, args.count)
        print_combinations(combinations, args.strategy)
    
    # Disclaimer
    print("\n" + "="*70)
    print("⚠️  IMPORTANT - CITEȘTE CU ATENȚIE")
    print("="*70)
    print("""
Aceste combinații sunt generate pe bază de statistici și algoritmi,
DAR nu pot prezice rezultatele viitoare!

Fiecare extragere Loto 5/40 este:
• Complet independentă de extragerile anterioare
• Bazată pe extragere FIZICĂ cu bile
• Imposibil de prezis cu orice algoritm

Șansele de câștig sunt EXACT ACELEAȘI pentru:
✓ Combinația generată "inteligent" de acest program
✓ Combinația aleasă complet random
✓ Combinația ta preferată (ziua de naștere, etc.)

Probabilitatea de a câștiga:
  • Categoria I (5/5 din primele 5): 1 în 658.008
  • Categoria II (5/6 din toate 6): 1 în 3.838.380

Folosește aceste combinații doar pentru distracție!
Joacă responsabil!
    """)


if __name__ == '__main__':
    main()
