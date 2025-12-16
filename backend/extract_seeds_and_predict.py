#!/usr/bin/env python3
"""
Extrage seed-urile din 2010-2025 pentru xorshift_simple
și generează predicția pentru următoarea extragere
"""

import json
import sys
from advanced_rng_library import create_rng, generate_numbers
from advanced_pattern_finder import AdvancedPatternFinder

LOTTERY = "5-40"  # Schimbă cu loteria ta
DATA_FILE = f"{LOTTERY}_data.json"

print("="*70)
print("EXTRAGERE SEEDS ȘI GENERARE PREDICȚIE")
print("="*70)
print()

# Încarcă datele complete
with open(DATA_FILE, 'r') as f:
    all_data = json.load(f)

# Pentru fiecare an 2010-2025, găsește best seed
print("Căutare seeds pentru fiecare an (2010-2025)...\n")

seeds_sequence = []

for year in range(2010, 2026):
    print(f"Anul {year}...", end=" ", flush=True)
    
    # Filtrează draws pentru acest an
    year_draws = [d for d in all_data['draws'] if d['year'] == year]
    
    if not year_draws:
        print("Nu există date")
        continue
    
    # Găsește best seed pentru acest an
    best_seed = None
    best_matches = 0
    
    # Test rapid - sample de seeds
    import random
    test_seeds = random.sample(range(1, 10000000), 50000)  # 50k seeds
    
    for seed in test_seeds:
        try:
            rng = create_rng('xorshift_simple', seed)
            matches = 0
            
            for draw in year_draws:
                target = set(draw['numbers_sorted'])
                
                # Generează numere
                if LOTTERY == 'joker':
                    # Pentru Joker (5/45 + 1/20)
                    nums = generate_numbers(rng, 5, 1, 45)
                    nums.append(generate_numbers(rng, 1, 1, 20)[0])
                else:
                    # Pentru 5/40 sau 6/49
                    config = all_data['config']
                    nums = generate_numbers(rng, config['numbers_to_draw'], 
                                           config['min_number'], config['max_number'])
                
                generated = set(nums)
                match_count = len(target & generated)
                
                if match_count >= 3:  # Min 3 matches
                    matches += 1
            
            if matches > best_matches:
                best_matches = matches
                best_seed = seed
        except:
            continue
    
    if best_seed:
        success_rate = best_matches / len(year_draws) if year_draws else 0
        print(f"Best seed: {best_seed:8d} | Success: {success_rate:.1%} ({best_matches}/{len(year_draws)})")
        seeds_sequence.append({
            'year': year,
            'seed': best_seed,
            'success_rate': success_rate,
            'matches': best_matches,
            'total': len(year_draws)
        })
    else:
        print("Nu s-a găsit seed bun")

print("\n" + "="*70)
print("SEEDS SEQUENCE")
print("="*70 + "\n")

if len(seeds_sequence) == 0:
    print("❌ Nu s-au găsit seeds cu success rate bun!")
    sys.exit(1)

# Print seeds
for s in seeds_sequence:
    marker = "🔥" if s['success_rate'] >= 0.7 else "⚠️" if s['success_rate'] >= 0.5 else "  "
    print(f"{marker} {s['year']}: Seed {s['seed']:10d} | {s['success_rate']:5.1%} ({s['matches']}/{s['total']})")

# Analizează pattern în seeds
print("\n" + "="*70)
print("ANALIZA PATTERN-ULUI ÎN SEEDS")
print("="*70 + "\n")

if len(seeds_sequence) < 3:
    print("⚠️ Prea puține seeds pentru pattern analysis")
    sys.exit(0)

seeds_only = [s['seed'] for s in seeds_sequence]

print(f"Seeds: {seeds_only}\n")

# Folosește AdvancedPatternFinder
pattern_finder = AdvancedPatternFinder(seeds_only)
patterns = pattern_finder.find_all_patterns()

if patterns:
    print("✅ PATTERNS GĂSITE!\n")
    
    for i, p in enumerate(patterns, 1):
        print(f"{i}. Pattern Type: {p['type']}")
        print(f"   Formula: {p['formula']}")
        print(f"   R²: {p['r_squared']:.4f}")
        print(f"   Next Seed: {p['next_seed']:,}")
        print()
    
    # Folosește best pattern pentru predicție
    best_pattern = patterns[0]
    next_seed = best_pattern['next_seed']
    
    print("="*70)
    print("🔮 PREDICȚIE PENTRU URMĂTOAREA EXTRAGERE")
    print("="*70 + "\n")
    
    print(f"Best Pattern: {best_pattern['formula']}")
    print(f"R² Score: {best_pattern['r_squared']:.4f}")
    print(f"Next Seed: {next_seed:,}\n")
    
    # Generează predicție
    rng = create_rng('xorshift_simple', next_seed)
    
    if LOTTERY == 'joker':
        predicted_nums = generate_numbers(rng, 5, 1, 45)
        joker_num = generate_numbers(rng, 1, 1, 20)[0]
        predicted_nums.append(joker_num)
        print(f"🎲 PREDICȚIE:")
        print(f"   Numere principale (5/45): {predicted_nums[:5]}")
        print(f"   Număr Joker (1/20): {joker_num}")
    else:
        config = all_data['config']
        predicted_nums = generate_numbers(rng, config['numbers_to_draw'],
                                         config['min_number'], config['max_number'])
        print(f"🎲 PREDICȚIE: {predicted_nums}")
    
    print(f"\nConfidence: {best_pattern['r_squared'] * 100:.1f}%")
    
    # Salvează predicția
    prediction_data = {
        'lottery': LOTTERY,
        'rng': 'xorshift_simple',
        'pattern': best_pattern,
        'next_seed': next_seed,
        'prediction': predicted_nums,
        'seeds_sequence': seeds_sequence
    }
    
    with open('PREDICTION_OUTPUT.json', 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    print("\n✅ Predicție salvată în: PREDICTION_OUTPUT.json")
    
else:
    print("❌ NU s-au găsit patterns matematice în seeds")
    print("Seeds variază aleatoriu\n")
    
    # Totuși, generează predicții probabilistice
    print("="*70)
    print("PREDICȚII PROBABILISTICE (fără pattern)")
    print("="*70 + "\n")
    
    import numpy as np
    
    # Median seed
    median_seed = int(np.median(seeds_only))
    print(f"1. Bazat pe Median Seed: {median_seed:,}")
    rng = create_rng('xorshift_simple', median_seed)
    config = all_data['config']
    pred = generate_numbers(rng, config['numbers_to_draw'],
                           config['min_number'], config['max_number'])
    print(f"   Predicție: {pred}\n")
    
    # Average seed
    avg_seed = int(np.mean(seeds_only))
    print(f"2. Bazat pe Average Seed: {avg_seed:,}")
    rng = create_rng('xorshift_simple', avg_seed)
    pred = generate_numbers(rng, config['numbers_to_draw'],
                           config['min_number'], config['max_number'])
    print(f"   Predicție: {pred}\n")
    
    # Recent trend (ultimii 3)
    if len(seeds_only) >= 3:
        recent_avg = int(np.mean(seeds_only[-3:]))
        print(f"3. Bazat pe Recent Trend: {recent_avg:,}")
        rng = create_rng('xorshift_simple', recent_avg)
        pred = generate_numbers(rng, config['numbers_to_draw'],
                               config['min_number'], config['max_number'])
        print(f"   Predicție: {pred}\n")

print("\n" + "="*70)
print("✅ ANALIZĂ COMPLETĂ!")
print("="*70)
