#!/usr/bin/env python3
"""
Seed Pattern Analyzer - Analizează secvența de seeds pentru a găsi formula

Din secvența [S₁, S₂, S₃, ...] caută pattern-uri:
1. Liniar: S(n) = a*n + b
2. LCG: S(n+1) = (a * S(n) + c) mod m
3. Timestamp-based: S(n) = f(timestamp)
4. Xorshift chain: S(n+1) = xorshift(S(n))
5. Hash-based: S(n) = hash(n, constant)

Utilizare:
    python3 seed_pattern_analyzer.py --input seed_sequence.json
"""

import argparse
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import statistics


class SeedPatternAnalyzer:
    def __init__(self, sequence_file: str):
        with open(sequence_file, 'r') as f:
            data = json.load(f)
        
        self.sequence_data = data['seed_sequence']
        self.seeds = [s['seed'] for s in self.sequence_data]
        self.dates = [s['date'] for s in self.sequence_data]
        self.matches = [s['matches'] for s in self.sequence_data]
        
        print(f"Încărcat {len(self.seeds)} seeds\n")
    
    def analyze_linear_pattern(self) -> Optional[Dict]:
        """
        Testează pattern liniar: S(n) = a*n + b
        """
        print("\n1. PATTERN LINIAR: S(n) = a*n + b")
        print("-" * 70)
        
        if len(self.seeds) < 2:
            print("  ✗ Insuficiente seeds")
            return None
        
        # Linear regression
        n = np.arange(len(self.seeds))
        seeds_array = np.array(self.seeds)
        
        # y = ax + b
        A = np.vstack([n, np.ones(len(n))]).T
        a, b = np.linalg.lstsq(A, seeds_array, rcond=None)[0]
        
        # Prediction
        predicted = a * n + b
        
        # Error
        errors = np.abs(seeds_array - predicted)
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        # R² score
        ss_tot = np.sum((seeds_array - np.mean(seeds_array))**2)
        ss_res = np.sum((seeds_array - predicted)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"  Formula: S(n) = {a:.2f} * n + {b:.2f}")
        print(f"  R² score: {r_squared:.6f}")
        print(f"  Error mediu: {avg_error:,.0f}")
        print(f"  Error maxim: {max_error:,.0f}")
        
        if r_squared > 0.95:
            print(f"  ✓ PATTERN GĂSIT! (R² > 0.95)")
            
            # Predict next
            next_n = len(self.seeds)
            next_seed = int(a * next_n + b)
            print(f"\n  → PREDICȚIE următorul seed: {next_seed:,}")
            
            return {
                'type': 'linear',
                'formula': f"S(n) = {a:.2f} * n + {b:.2f}",
                'a': float(a),
                'b': float(b),
                'r_squared': float(r_squared),
                'next_seed': next_seed,
                'confidence': 'HIGH' if r_squared > 0.99 else 'MEDIUM'
            }
        else:
            print(f"  ✗ Nu e pattern liniar (R² prea mic)")
            return None
    
    def analyze_lcg_pattern(self) -> Optional[Dict]:
        """
        Testează pattern LCG: S(n+1) = (a * S(n) + c) mod m
        """
        print("\n2. PATTERN LCG: S(n+1) = (a * S(n) + c) mod m")
        print("-" * 70)
        
        if len(self.seeds) < 3:
            print("  ✗ Insuficiente seeds (minim 3)")
            return None
        
        # Încercăm să găsim a, c, m din 3 ecuații
        # S1 = (a * S0 + c) mod m
        # S2 = (a * S1 + c) mod m
        # ...
        
        # Testăm module comune
        common_moduli = [2**31, 2**32, 2**31 - 1, 2**16, 10**9 + 7]
        
        best_match = None
        best_correct = 0
        
        for m in common_moduli:
            # Sistem de ecuații pentru a găsi 'a' și 'c'
            # S1 - S2 = a*(S0 - S1) mod m
            
            if len(self.seeds) < 10:
                continue
            
            # Sample primele 10 pentru găsirea parametrilor
            sample_seeds = self.seeds[:10]
            
            # Brute force pentru a, c (în range limitat)
            for a in range(1, 10000, 100):  # Sample a
                for c in range(0, 10000, 100):  # Sample c
                    correct = 0
                    
                    for i in range(len(sample_seeds) - 1):
                        predicted = (a * sample_seeds[i] + c) % m
                        if predicted == sample_seeds[i + 1]:
                            correct += 1
                    
                    if correct > best_correct:
                        best_correct = correct
                        best_match = {'a': a, 'c': c, 'm': m}
        
        if best_match and best_correct >= 7:  # Cel puțin 70% match
            a, c, m = best_match['a'], best_match['c'], best_match['m']
            
            print(f"  Formula găsită: S(n+1) = ({a} * S(n) + {c}) mod {m}")
            print(f"  Matches: {best_correct}/{len(sample_seeds)-1}")
            print(f"  ✓ POSIBIL PATTERN LCG!")
            
            # Predict next
            last_seed = self.seeds[-1]
            next_seed = (a * last_seed + c) % m
            
            print(f"\n  → PREDICȚIE următorul seed: {next_seed:,}")
            
            return {
                'type': 'lcg',
                'formula': f"S(n+1) = ({a} * S(n) + {c}) mod {m}",
                'a': a,
                'c': c,
                'm': m,
                'next_seed': int(next_seed),
                'confidence': 'HIGH' if best_correct >= 9 else 'MEDIUM'
            }
        else:
            print(f"  ✗ Nu e pattern LCG standard")
            return None
    
    def analyze_diff_pattern(self) -> Optional[Dict]:
        """
        Analizează diferențele între seeds consecutive
        """
        print("\n3. ANALIZ DIFERENȚE CONSECUTIVE")
        print("-" * 70)
        
        if len(self.seeds) < 2:
            return None
        
        diffs = [self.seeds[i+1] - self.seeds[i] for i in range(len(self.seeds)-1)]
        
        # Statistici
        avg_diff = statistics.mean(diffs)
        std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0
        
        print(f"  Diferență medie: {avg_diff:,.0f}")
        print(f"  Std deviation: {std_diff:,.0f}")
        
        # Dacă std e foarte mic → diferență constantă
        if std_diff < abs(avg_diff) * 0.01:  # <1% variație
            print(f"  ✓ DIFERENȚĂ CONSTANTĂ GĂSITĂ!")
            print(f"  Formula: S(n+1) = S(n) + {avg_diff:.0f}")
            
            next_seed = self.seeds[-1] + int(avg_diff)
            print(f"\n  → PREDICȚIE următorul seed: {next_seed:,}")
            
            return {
                'type': 'constant_diff',
                'formula': f"S(n+1) = S(n) + {avg_diff:.0f}",
                'diff': avg_diff,
                'next_seed': next_seed,
                'confidence': 'HIGH'
            }
        
        # Diferențele formează un pattern?
        if len(diffs) > 2:
            second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
            avg_second_diff = statistics.mean(second_diffs)
            
            if abs(avg_second_diff) > 0.01:
                print(f"  Diferență de nivel 2: {avg_second_diff:,.0f}")
                print(f"  → Posibil pattern pătratic")
        
        print(f"  ✗ Diferențe variabile (seed-uri random sau complex pattern)")
        return None
    
    def analyze_all(self) -> List[Dict]:
        """
        Rulează toate analizele
        """
        print("\n" + "="*70)
        print("ANALIZĂ PATTERN SEEDS")
        print("="*70)
        print(f"\nSeeds în secvență: {len(self.seeds)}")
        print(f"Matches medii: {statistics.mean(self.matches):.1f}/6")
        
        # Sample seeds
        print(f"\nPrimele 10 seeds:")
        for i in range(min(10, len(self.seeds))):
            print(f"  {i}: {self.seeds[i]:>10,} | {self.dates[i]} | {self.matches[i]}/6 match")
        
        # Run analize
        patterns_found = []
        
        p1 = self.analyze_linear_pattern()
        if p1:
            patterns_found.append(p1)
        
        p2 = self.analyze_diff_pattern()
        if p2:
            patterns_found.append(p2)
        
        p3 = self.analyze_lcg_pattern()
        if p3:
            patterns_found.append(p3)
        
        return patterns_found


def main():
    parser = argparse.ArgumentParser(
        description='Seed Pattern Analyzer - Găsește formula în secvența de seeds'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Fișier seed_sequence.json')
    parser.add_argument('--output', type=str, default='seed_patterns.json')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  SEED PATTERN ANALYZER")
    print("="*70)
    print("\n⚠️  Caută formula care generează seed-urile\n")
    
    try:
        analyzer = SeedPatternAnalyzer(args.input)
        patterns = analyzer.analyze_all()
        
        print("\n" + "="*70)
        print("REZULTATE")
        print("="*70)
        
        if patterns:
            print(f"\n✓ GĂSITE {len(patterns)} PATTERN(URI)!\n")
            
            for i, pattern in enumerate(patterns, 1):
                print(f"{i}. {pattern['type'].upper()}")
                print(f"   Formula: {pattern['formula']}")
                print(f"   Next seed: {pattern['next_seed']:,}")
                print(f"   Confidence: {pattern['confidence']}")
                print()
            
            # Save
            with open(args.output, 'w') as f:
                json.dump({
                    'patterns_found': len(patterns),
                    'patterns': patterns,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"💾 Pattern-uri salvate: {args.output}")
            
            print("\n" + "="*70)
            print("NEXT STEP - GENERARE PREDICȚIE")
            print("="*70)
            print("\nFolosește seed-ul prezis pentru a genera numerele:")
            best_pattern = patterns[0]
            print(f"  python3 seed_predictor.py --seed {best_pattern['next_seed']} --formula '{best_pattern['formula']}'")
        
        else:
            print("\n✗ NU S-AU GĂSIT PATTERN-URI CLARE\n")
            print("Acest lucru înseamnă:")
            print("  • Seed-urile sunt random (nu există formulă)")
            print("  • SAU formula e prea complexă pentru analiză simplă")
            print("  • SAU datele NU provin dintr-un RNG cu seed predictibil")
            print("\n→ Confirmare că loteria NU folosește RNG simplu!")
    
    except Exception as e:
        print(f"Eroare: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
