#!/usr/bin/env python3
"""
Seed Predictor - Generează următoarea extragere din seed prezis

Folosește seed-ul prezis din formula și generează numerele câștigătoare.

Utilizare:
    python3 seed_predictor.py --seed 12345678 --formula "S(n+1) = S(n) + 1000"
    python3 seed_predictor.py --pattern-file seed_patterns.json
"""

import argparse
import json
from typing import List


class FastLCG:
    """LCG pentru generare numere din seed"""
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


def generate_from_seed(seed: int, rng_type: str = 'lcg') -> List[int]:
    """Generează 6 numere din seed"""
    rng = FastLCG(seed)
    return rng.generate_numbers(6, 1, 40)


def main():
    parser = argparse.ArgumentParser(
        description='Seed Predictor - Generează următoarea extragere din seed'
    )
    parser.add_argument('--seed', type=int, help='Seed de folosit')
    parser.add_argument('--formula', type=str, help='Formula folosită')
    parser.add_argument('--pattern-file', type=str, help='Fișier cu pattern-uri găsite')
    parser.add_argument('--rng', type=str, default='lcg', choices=['lcg', 'xorshift'])
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  SEED PREDICTOR - Generare Următoare Extragere")
    print("="*70)
    
    seeds_to_test = []
    
    # Load din pattern file
    if args.pattern_file:
        try:
            with open(args.pattern_file, 'r') as f:
                patterns_data = json.load(f)
            
            print(f"\nÎncărcat {patterns_data['patterns_found']} pattern(uri)\n")
            
            for i, pattern in enumerate(patterns_data['patterns'], 1):
                print(f"{i}. {pattern['type'].upper()}")
                print(f"   Formula: {pattern['formula']}")
                print(f"   Seed prezis: {pattern['next_seed']:,}")
                print(f"   Confidence: {pattern['confidence']}")
                
                seeds_to_test.append({
                    'seed': pattern['next_seed'],
                    'formula': pattern['formula'],
                    'type': pattern['type'],
                    'confidence': pattern['confidence']
                })
                print()
        
        except Exception as e:
            print(f"Eroare la citirea pattern file: {e}")
            return
    
    # Sau seed manual
    elif args.seed:
        seeds_to_test.append({
            'seed': args.seed,
            'formula': args.formula or "Manual",
            'type': 'manual',
            'confidence': 'UNKNOWN'
        })
    
    else:
        print("Eroare: Specifică --seed sau --pattern-file")
        return
    
    # Generează predicții
    print("\n" + "="*70)
    print("PREDICȚII GENERATE")
    print("="*70)
    
    for i, seed_info in enumerate(seeds_to_test, 1):
        print(f"\n{i}. Pattern: {seed_info['type'].upper()}")
        print(f"   Formula: {seed_info['formula']}")
        print(f"   Seed: {seed_info['seed']:,}")
        print(f"   Confidence: {seed_info['confidence']}")
        print()
        
        # Generează numere
        numbers = generate_from_seed(seed_info['seed'], args.rng)
        
        print(f"   🎲 PREDICȚIE URMĂTOARE EXTRAGERE:")
        print(f"   ╔═══════════════════════════════════╗")
        print(f"   ║  {' - '.join([f'{n:2d}' for n in numbers])}  ║")
        print(f"   ╚═══════════════════════════════════╝")
        print()
        
        # Salvează
        output = {
            'seed': seed_info['seed'],
            'formula': seed_info['formula'],
            'predicted_numbers': numbers,
            'sorted': numbers,
            'pattern_type': seed_info['type'],
            'confidence': seed_info['confidence']
        }
        
        output_file = f"prediction_{seed_info['seed']}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"   💾 Salvat: {output_file}")
    
    print("\n" + "="*70)
    print("⚠️  IMPORTANT - DISCLAIMER")
    print("="*70)
    print("""
Aceste predicții sunt bazate pe:
1. Seed-uri "găsite" pentru extrageri istorice
2. Pattern-uri detectate în secvența de seeds
3. Presupunerea că există o formulă

REALITATEA:
• Loteriile folosesc extragere FIZICĂ cu bile
• NU există seed-uri sau formule reale
• Orice "pattern" găsit este coincidență statistică
• Aceste predicții NU au nicio valoare reală

Scopul acestui exercițiu:
→ Demonstrează EXPERIMENTAL că nu există pattern
→ "Predicțiile" vor fi GREȘITE
→ Confirmă că datele sunt aleatorii

NU folosi aceste predicții pentru jocuri reale!
    """)
    
    print("\n" + "="*70)
    print("VERIFICARE PREDICȚIE")
    print("="*70)
    print("""
Când următoarea extragere va avea loc:

1. Compară rezultatul real cu predicția
2. Numără match-urile (probabil 0-2 din 6)
3. Observă că predicția NU funcționează
4. Concluzie: NU există formulă → Datele sunt aleatorii!

Pentru verificare automată:
  python3 verify_prediction.py --prediction prediction_XXXXX.json --actual "1,5,12,23,34,39"
    """)


if __name__ == '__main__':
    main()
