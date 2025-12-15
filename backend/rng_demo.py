#!/usr/bin/env python3
"""
Demonstrație educațională: Reverse Engineering RNG (Xorshift32)

Acest script demonstrează conceptul de "seed finding" pentru generatoare
pseudo-random SIMPLE, similar cu exemplele din video-urile despre jocuri.

⚠️ IMPORTANT: Această tehnică funcționează DOAR pentru:
   - Jocuri video simple
   - Generatoare pseudo-random neprotejate
   - Aplicații cu RNG slab

❌ NU funcționează pentru:
   - Loterii oficiale (folosesc extragere FIZICĂ cu bile)
   - Sisteme cu RNG criptografic
   - Orice sistem securizat corespunzător

Utilizare:
    python3 rng_demo.py --demo
    python3 rng_demo.py --find-seed 12345678
"""

import argparse
import random
from typing import List, Optional


class Xorshift32:
    """
    Implementare simplă Xorshift32 - un RNG pseudo-random FOARTE simplu
    folosit în demonstrații și jocuri simple.
    """
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF  # 32-bit state
        if self.state == 0:
            self.state = 1  # Evită starea 0
    
    def next(self) -> int:
        """
        Generează următorul număr pseudo-random
        Algoritmul Xorshift32 clasic
        """
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x
    
    def next_in_range(self, min_val: int, max_val: int) -> int:
        """
        Generează un număr în intervalul [min_val, max_val]
        """
        return min_val + (self.next() % (max_val - min_val + 1))


class XorshiftReverse:
    """
    Demonstrație de inversare Xorshift32
    """
    
    @staticmethod
    def inverse_xor_left_shift(value: int, shift: int) -> int:
        """
        Inversează operația: x ^= (x << shift)
        """
        mask = 0xFFFFFFFF
        result = value
        
        for i in range(32 // shift + 1):
            result = value ^ ((result << shift) & mask)
        
        return result
    
    @staticmethod
    def inverse_xor_right_shift(value: int, shift: int) -> int:
        """
        Inversează operația: x ^= (x >> shift)
        """
        mask = 0xFFFFFFFF
        result = value
        
        for i in range(32 // shift + 1):
            result = value ^ ((result >> shift) & mask)
        
        return result
    
    @staticmethod
    def reverse_step(current_state: int) -> int:
        """
        Inversează un pas al Xorshift32
        """
        # Inversăm în ordine inversă:
        # Original: x ^= (x << 13); x ^= (x >> 17); x ^= (x << 5)
        
        x = current_state
        x = XorshiftReverse.inverse_xor_left_shift(x, 5)
        x = XorshiftReverse.inverse_xor_right_shift(x, 17)
        x = XorshiftReverse.inverse_xor_left_shift(x, 13)
        
        return x & 0xFFFFFFFF


def demo_xorshift():
    """
    Demonstrație: Generare și reversare Xorshift32
    """
    print("\n" + "="*70)
    print("DEMONSTRAȚIE: Reverse Engineering Xorshift32 RNG")
    print("="*70)
    
    # 1. Generăm o secvență cu un seed cunoscut
    seed = 12345
    rng = Xorshift32(seed)
    
    print(f"\n1. Generăm o secvență cu seed = {seed}")
    print("-" * 70)
    
    sequence = []
    for i in range(5):
        value = rng.next()
        sequence.append(value)
        print(f"  Pas {i+1}: {value:10d} (0x{value:08X})")
    
    # 2. Acum "uităm" seed-ul și încercăm să-l recuperăm
    print("\n2. Recuperăm seed-ul din secvență (reverse engineering)")
    print("-" * 70)
    print("  Presupunem că știm doar ultimul output...")
    
    last_state = sequence[-1]
    print(f"  Ultimul output: {last_state}")
    
    # Inversăm toți pașii
    current = last_state
    print("\n  Inversare pas cu pas:")
    for i in range(len(sequence) - 1, 0, -1):
        prev = XorshiftReverse.reverse_step(current)
        print(f"    Pas {i} → Pas {i-1}: {prev:10d}")
        current = prev
    
    # Seed-ul recuperat
    recovered_seed = XorshiftReverse.reverse_step(sequence[0])
    print(f"\n  Seed ORIGINAL:   {seed}")
    print(f"  Seed RECUPERAT:  {recovered_seed}")
    
    if recovered_seed == seed:
        print("  ✓ SUCCES: Am recuperat seed-ul!")
    else:
        print("  ⚠ Seed-ul diferă (dar algoritmul funcționează pentru forward prediction)")
    
    # 3. Demonstrație seed finding prin brute force
    print("\n3. Simulare 'Seed Finding' (ca în video-urile despre jocuri)")
    print("-" * 70)
    print("  Scenariul: Știm primele 3 numere generate, căutăm seed-ul...")
    
    target_sequence = sequence[:3]
    print(f"  Secvența țintă: {target_sequence}")
    
    print("\n  Căutare seed (simulăm o căutare limitată)...")
    found_seed = None
    
    # Căutăm în intervalul [10000, 15000]
    for test_seed in range(10000, 15001):
        test_rng = Xorshift32(test_seed)
        match = True
        
        for expected in target_sequence:
            if test_rng.next() != expected:
                match = False
                break
        
        if match:
            found_seed = test_seed
            break
    
    if found_seed:
        print(f"  ✓ GĂSIT! Seed-ul este: {found_seed}")
        print(f"  Verificare: Generăm următoarele valori...")
        
        verify_rng = Xorshift32(found_seed)
        for _ in range(3):
            verify_rng.next()  # Skip primele 3
        
        print(f"  Următoarele predicții: {[verify_rng.next() for _ in range(3)]}")
        print(f"  Valorile reale:        {sequence[3:]}")
    else:
        print("  Nu s-a găsit seed-ul în intervalul căutat.")


def simulate_lotto_impossibility():
    """
    Demonstrație de ce tehnicile RNG NU funcționează pentru loterii
    """
    print("\n" + "="*70)
    print("DE CE NU FUNCȚIONEAZĂ LA LOTERII REALE?")
    print("="*70)
    
    print("""
1. LOTERIA FOLOSEȘTE EXTRAGERE FIZICĂ:
   • Bile fizice în mașini mecanice
   • Fără seed, fără algoritm software
   • Fiecare extragere este COMPLET independentă
   • Influențată de: temperatură, umiditate, forța de amestec, etc.

2. CHIAR DACĂ AR FI SOFTWARE (nu este!):
   • Ar folosi RNG criptografic (ex: /dev/urandom, hardware RNG)
   • Seed-uri de 256+ biți
   • Imposibil de inversat sau brute-force

3. DIFERENȚA FUNDAMENTALĂ:
   • Joc video simplu: Xorshift32 = 4.3 miliarde de seed-uri posibile
   • RNG modern: 2^256 = mai mult decât atomi în univers
   • Loterie fizică: ∞ (infinit) - nu există seed

4. DE CE FUNCȚIONEAZĂ ÎN VIDEO-URI:
   • Jocuri vechi (ex: Minesweeper, Pokemon) folosesc RNG simplu
   • Seed-uri mici (16-32 biți)
   • Fără protecție criptografică
   • Scopul era performanță, nu securitate

    """)
    
    print("-" * 70)
    print("CONCLUZIE:")
    print("-" * 70)
    print("""
Tehnicile de "seed finding" sunt fascinante pentru:
✓ Reverse engineering jocuri video
✓ Speedrunning / Tool-assisted speedruns
✓ Învățarea conceptelor de RNG
✓ Securitate informatică (găsirea vulnerabilităților)

DAR sunt COMPLET INUTILE pentru:
✗ Loterii oficiale
✗ Cazinouri online reglementate  
✗ Sisteme cu securitate criptografică
✗ Orice aplicație de gambling licențiată

Numai fiți precauți cu promisiunile de "seed-uri magice" pentru loto!
    """)


def main():
    parser = argparse.ArgumentParser(
        description='Demonstrație educațională RNG reverse engineering',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Rulează demonstrația completă'
    )
    parser.add_argument(
        '--find-seed',
        type=int,
        metavar='NUM',
        help='Caută seed care generează numărul dat'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        demo_xorshift()
        simulate_lotto_impossibility()
    elif args.find_seed:
        print(f"Căutare seed pentru output: {args.find_seed}...")
        # Simplificat - căută în primele 100k
        for seed in range(1, 100000):
            rng = Xorshift32(seed)
            if rng.next() == args.find_seed:
                print(f"✓ Găsit seed: {seed}")
                break
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
