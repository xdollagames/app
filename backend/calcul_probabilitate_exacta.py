#!/usr/bin/env python3
"""
CALCULUL PROBABILITĂȚII: Câte seeds trebuie testate pentru a găsi ordinea EXACTĂ?
"""

from advanced_rng_library import create_rng, generate_numbers
import time
from collections import defaultdict

target_exact = [6, 27, 9, 31, 4, 11]
target_sorted = sorted(target_exact)

print("=" * 80)
print("🎯 CĂUTARE EXHAUSTIVĂ: Seeds care generează ordinea EXACTĂ")
print("=" * 80)
print(f"\nTarget exact:  {target_exact}")
print(f"Target sorted: {target_sorted}")

# Testăm un range mare de seeds
test_range = 5_000_000  # 5 milioane
rng_type = 'lcg_minstd'

seeds_match_sorted = []
seeds_match_exact = []
ordini_gasite = defaultdict(int)  # Contorizăm fiecare ordine găsită

print(f"\n🔍 Testăm {test_range:,} seeds pentru {rng_type}...")
print(f"⏱️  Aceasta va dura ~30-60 secunde...")
start = time.time()

checkpoint = test_range // 10
for seed in range(test_range):
    if seed > 0 and seed % checkpoint == 0:
        elapsed = time.time() - start
        progress = (seed / test_range) * 100
        print(f"   Progress: {progress:.0f}% ({seed:,}/{test_range:,}) - "
              f"Sorted: {len(seeds_match_sorted)}, Exact: {len(seeds_match_exact)}")
    
    try:
        rng = create_rng(rng_type, seed)
        generated = generate_numbers(rng, 6, 1, 40)
        
        # Verificare cu SORTED
        if sorted(generated) == target_sorted:
            seeds_match_sorted.append((seed, generated))
            # Salvăm ordinea pentru analiză
            ordini_gasite[tuple(generated)] += 1
        
        # Verificare EXACT
        if generated == target_exact:
            seeds_match_exact.append((seed, generated))
    except:
        continue

elapsed = time.time() - start

print("\n" + "=" * 80)
print(f"✅ CĂUTARE COMPLETĂ!")
print(f"⏱️  Timp total: {elapsed:.1f} secunde")
print(f"📊 Seeds testate: {test_range:,}")
print("=" * 80)

print(f"\n📊 REZULTATE:")
print(f"\n1️⃣  Cu SORTED (orice ordine a setului {target_sorted}):")
print(f"   ✅ Seeds găsite: {len(seeds_match_sorted)}")

if seeds_match_sorted:
    print(f"\n   Primele 10 seeds:")
    for i, (seed, gen) in enumerate(seeds_match_sorted[:10]):
        print(f"      #{i+1}. Seed {seed:9,} → {gen}")
    
    if len(seeds_match_sorted) > 10:
        print(f"      ... și încă {len(seeds_match_sorted) - 10} seeds")

print(f"\n2️⃣  Cu ORDINEA EXACTĂ {target_exact}:")
print(f"   ✅ Seeds găsite: {len(seeds_match_exact)}")

if seeds_match_exact:
    print(f"\n   Seeds care generează EXACT ordinea fizică:")
    for seed, gen in seeds_match_exact:
        print(f"      🎯 Seed {seed:9,} → {gen}")
else:
    print(f"   ❌ NICIUN seed găsit care să genereze exact ordinea fizică!")

# Analizăm distribuția ordinilor
if ordini_gasite:
    print(f"\n3️⃣  ANALIZA ORDINILOR generate de seeds-urile găsite:")
    print(f"   📊 Seeds care produc setul corect: {len(seeds_match_sorted)}")
    print(f"   📊 Ordini diferite găsite: {len(ordini_gasite)}")
    
    print(f"\n   Top 10 ordini cele mai frecvente:")
    ordini_sorted = sorted(ordini_gasite.items(), key=lambda x: x[1], reverse=True)
    for i, (ordine, count) in enumerate(ordini_sorted[:10]):
        is_target = list(ordine) == target_exact
        marker = "🎯" if is_target else "  "
        print(f"      {marker} #{i+1}. {list(ordine)} → {count} seeds")

# Calculăm probabilitățile
print("\n" + "=" * 80)
print("📈 STATISTICI ȘI PROBABILITĂȚI:")
print("=" * 80)

if len(seeds_match_sorted) > 0:
    ratio = len(seeds_match_sorted) / test_range
    print(f"\n🔢 Probabilitate de a găsi setul corect (sorted):")
    print(f"   {len(seeds_match_sorted)}/{test_range:,} = {ratio:.6f} = 1/{int(1/ratio):,}")
    
    if len(seeds_match_exact) > 0:
        ratio_exact = len(seeds_match_exact) / test_range
        print(f"\n🔢 Probabilitate de a găsi ordinea EXACTĂ:")
        print(f"   {len(seeds_match_exact)}/{test_range:,} = {ratio_exact:.6f} = 1/{int(1/ratio_exact):,}")
        
        # Raportul între cele două
        improvement = len(seeds_match_sorted) / len(seeds_match_exact)
        print(f"\n📊 SORTED găsește de {improvement:.1f}x MAI MULTE seeds decât EXACT!")
    else:
        print(f"\n🔢 Probabilitate de a găsi ordinea EXACTĂ:")
        print(f"   0/{test_range:,} → MAI PUȚIN de 1/{test_range:,}")
        print(f"\n💡 Ar fi nevoie de MAI MULT de {test_range:,} încercări!")
        
        if len(seeds_match_sorted) > 0 and len(ordini_gasite) > 0:
            # Estimare: dacă am găsit N seeds cu setul corect în M ordini diferite
            # și presupunem distribuție uniformă, probabilitatea pentru o ordine specifică = 1/M
            estimated_tries = test_range * len(ordini_gasite)
            print(f"\n📊 ESTIMARE (dacă ordinile sunt distribuite uniform):")
            print(f"   Am găsit {len(seeds_match_sorted)} seeds în {len(ordini_gasite)} ordini diferite")
            print(f"   Pentru a găsi o ordine specifică ar fi nevoie de ~{estimated_tries:,} încercări")
            print(f"   (în medie 1 din {len(ordini_gasite)} seeds cu setul corect are ordinea dorită)")

print("\n" + "=" * 80)
print("🎓 CONCLUZIE:")
print("=" * 80)
print(f"Cu SORTED găsim MULT MAI UȘOR seeds-uri valide!")
print(f"Ordinea fizică EXACTĂ este EXTREM de rară sau inexistentă în spațiul de căutare.")
