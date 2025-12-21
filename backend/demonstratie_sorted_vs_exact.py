#!/usr/bin/env python3
"""
Demonstrație: Sorted vs. Ordinea Exactă
Care abordare găsește MAI MULTE seed-uri?
"""

from advanced_rng_library import create_rng, generate_numbers
import time

# Extragerea target din JSON (ordinea fizică)
target = [6, 27, 9, 31, 4, 11]
target_sorted = sorted(target)

print("=" * 70)
print("🎯 TARGET (ordinea fizică din JSON):", target)
print("🔢 TARGET SORTED:", target_sorted)
print("=" * 70)

# Testăm primele 100,000 seeds pentru LCG_MINSTD
test_range = 100000
rng_type = 'lcg_minstd'

seeds_match_sorted = []
seeds_match_exact = []

print(f"\n🔍 Testăm primele {test_range:,} seeds pentru {rng_type}...\n")
start = time.time()

for seed in range(test_range):
    try:
        rng = create_rng(rng_type, seed)
        generated = generate_numbers(rng, 6, 1, 40)
        
        # Verificare cu SORTED (comparăm setul)
        if sorted(generated) == target_sorted:
            seeds_match_sorted.append((seed, generated))
        
        # Verificare EXACT (comparăm ordinea)
        if generated == target:
            seeds_match_exact.append((seed, generated))
    except:
        continue

elapsed = time.time() - start

print("=" * 70)
print(f"⏱️  Timp de căutare: {elapsed:.2f} secunde")
print("=" * 70)

print(f"\n📊 REZULTATE:")
print(f"\n1️⃣  Cu SORTED (comparăm SETUL de numere):")
print(f"   ✅ Seeds găsite: {len(seeds_match_sorted)}")
if seeds_match_sorted:
    print(f"   Primele 5 exemple:")
    for seed, gen in seeds_match_sorted[:5]:
        print(f"      Seed {seed:7d} → {gen}")

print(f"\n2️⃣  Cu ORDINEA EXACTĂ (comparăm ordinea fizică):")
print(f"   ✅ Seeds găsite: {len(seeds_match_exact)}")
if seeds_match_exact:
    print(f"   Exemple:")
    for seed, gen in seeds_match_exact:
        print(f"      Seed {seed:7d} → {gen}")
else:
    print(f"   ❌ NICIUN seed găsit care să genereze exact ordinea fizică!")

print("\n" + "=" * 70)
print("🎓 CONCLUZIE:")
print("=" * 70)

if len(seeds_match_sorted) > len(seeds_match_exact):
    ratio = len(seeds_match_sorted) / max(len(seeds_match_exact), 1)
    print(f"✅ Cu SORTED găsim {len(seeds_match_sorted)} seeds")
    print(f"❌ Cu EXACT găsim {len(seeds_match_exact)} seeds")
    if len(seeds_match_exact) == 0:
        print(f"\n💡 SORTED găsește seeds, EXACT nu găsește NIMIC!")
        print(f"   De aceea sistemul nu funcționa fără sorted()!")
    else:
        print(f"\n💡 SORTED găsește de {ratio:.1f}x MAI MULTE seed-uri!")
    
    print(f"\n🔬 EXPLICAȚIE:")
    print(f"   - SORTED: Orice ordine de generare RNG care conține {target_sorted} → MATCH")
    print(f"   - EXACT: Doar RNG care generează EXACT {target} → MATCH (extrem de rar!)")
    print(f"\n   Pentru un RNG să genereze EXACT ordinea fizică aleatoare e aproape imposibil!")
    print(f"   Ordinea RNG este DETERMINISTĂ, ordinea fizică este ALEATOARE.")
