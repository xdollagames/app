#!/usr/bin/env python3
"""
TESTARE EXTREMĂ: Căutare în ÎNTREG range-ul optim (4,000,000 seeds)
pentru a vedea dacă VREODATĂ găsim ordinea exactă
"""

from advanced_rng_library import create_rng, generate_numbers
import time

target_exact = [6, 27, 9, 31, 4, 11]
target_sorted = sorted(target_exact)

print("=" * 80)
print("🎯 TESTARE COMPLETĂ: TOT RANGE-UL OPTIMIZAT (4,000,000 seeds)")
print("=" * 80)
print(f"\nTarget exact:  {target_exact}")
print(f"Target sorted: {target_sorted}")

# Testăm ÎNTREG range-ul optim pentru 5-40
test_range = 4_000_000  # Range-ul complet optimizat
rng_type = 'lcg_minstd'

seeds_with_set = []
seeds_with_exact = []
all_orders_found = {}

print(f"\n🔍 Testăm TOȚI cei {test_range:,} seeds pentru {rng_type}...")
print(f"⏱️  Aceasta va dura ~1-2 minute...")
print(f"\n📊 Progress:")
start = time.time()

checkpoint = test_range // 20
for seed in range(test_range):
    if seed > 0 and seed % checkpoint == 0:
        elapsed = time.time() - start
        progress = (seed / test_range) * 100
        speed = seed / elapsed if elapsed > 0 else 0
        eta = (test_range - seed) / speed if speed > 0 else 0
        print(f"   {progress:5.1f}% | Seeds cu set corect: {len(seeds_with_set):3d} | "
              f"Seeds cu ordine exactă: {len(seeds_with_exact):3d} | "
              f"ETA: {eta:.0f}s")
    
    try:
        rng = create_rng(rng_type, seed)
        generated = generate_numbers(rng, 6, 1, 40)
        
        # Verificare cu SORTED
        if sorted(generated) == target_sorted:
            seeds_with_set.append((seed, generated))
            order_key = tuple(generated)
            if order_key not in all_orders_found:
                all_orders_found[order_key] = []
            all_orders_found[order_key].append(seed)
        
        # Verificare EXACT
        if generated == target_exact:
            seeds_with_exact.append((seed, generated))
    except:
        continue

elapsed = time.time() - start

print("\n" + "=" * 80)
print(f"✅ CĂUTARE 100% EXHAUSTIVĂ COMPLETĂ!")
print(f"⏱️  Timp total: {elapsed:.1f} secunde ({elapsed/60:.1f} minute)")
print(f"📊 Seeds testate: {test_range:,} (TOT RANGE-UL OPTIMIZAT)")
print("=" * 80)

print(f"\n📊 REZULTATE FINALE:")
print(f"\n1️⃣  Seeds care generează SETUL {target_sorted} (în orice ordine):")
print(f"   ✅ Total găsite: {len(seeds_with_set)} seeds")

if seeds_with_set:
    print(f"\n   Toate seeds-urile găsite:")
    for i, (seed, gen) in enumerate(seeds_with_set):
        print(f"      #{i+1}. Seed {seed:9,} → {gen}")
else:
    print(f"   ❌ Niciun seed găsit!")

print(f"\n2️⃣  Seeds care generează ORDINEA EXACTĂ {target_exact}:")
print(f"   ✅ Total găsite: {len(seeds_with_exact)} seeds")

if seeds_with_exact:
    print(f"\n   🎯🎯🎯 GĂSIT SEED CU ORDINEA EXACTĂ! 🎯🎯🎯")
    for seed, gen in seeds_with_exact:
        print(f"      Seed {seed:9,} → {gen}")
else:
    print(f"   ❌ NICIUN seed nu generează ordinea fizică exactă!")
    print(f"   ❌ Am testat TOATE cele {test_range:,} seeds din range-ul optimizat!")

print(f"\n3️⃣  DISTRIBUȚIA ORDINILOR:")
print(f"   📊 Ordini diferite găsite: {len(all_orders_found)}")

if all_orders_found:
    print(f"\n   Toate ordinile găsite:")
    for i, (order, seed_list) in enumerate(sorted(all_orders_found.items())):
        is_target = list(order) == target_exact
        marker = "🎯 TARGET!" if is_target else ""
        print(f"      #{i+1}. {list(order)} → {len(seed_list)} seed(s) {marker}")
        for s in seed_list:
            print(f"           Seed: {s:,}")

# Statistici finale
print("\n" + "=" * 80)
print("📈 STATISTICI FINALE:")
print("=" * 80)

print(f"\n🔢 Probabilități în range-ul {test_range:,}:")
if len(seeds_with_set) > 0:
    prob_set = len(seeds_with_set) / test_range
    print(f"   Setul corect (sorted): {len(seeds_with_set)}/{test_range:,} = "
          f"{prob_set:.8f} = 1/{int(1/prob_set):,}")
else:
    print(f"   Setul corect: 0 seeds găsite")

if len(seeds_with_exact) > 0:
    prob_exact = len(seeds_with_exact) / test_range
    print(f"   Ordinea exactă: {len(seeds_with_exact)}/{test_range:,} = "
          f"{prob_exact:.8f} = 1/{int(1/prob_exact):,}")
    
    ratio = len(seeds_with_set) / len(seeds_with_exact)
    print(f"\n📊 SORTED găsește de {ratio:.1f}x MAI MULTE seeds!")
else:
    print(f"   Ordinea exactă: 0/{test_range:,}")
    print(f"\n   💡 Probabilitatea < 1/{test_range:,} = < 0.00000025")
    print(f"   💡 Ar fi nevoie de PESTE {test_range:,} încercări!")

print("\n" + "=" * 80)
print("🎓 CONCLUZIE FINALĂ:")
print("=" * 80)

if len(seeds_with_exact) == 0:
    print(f"\n✅ În TOT range-ul optimizat ({test_range:,} seeds):")
    print(f"   • Găsim {len(seeds_with_set)} seeds cu setul corect (SORTED)")
    print(f"   • Găsim 0 seeds cu ordinea exactă (FĂRĂ SORTED)")
    print(f"\n💡 Pentru a găsi ordinea fizică exactă [6, 27, 9, 31, 4, 11]")
    print(f"   ar fi probabil nevoie de ZECI DE MILIOANE de încercări!")
    print(f"\n🎯 DE ACEEA FOLOSIM SORTED - este singura metodă PRACTICĂ!")
else:
    print(f"\n🎯 Am găsit {len(seeds_with_exact)} seed(s) cu ordinea exactă!")
    print(f"   Dar tot am găsit {len(seeds_with_set)} seeds cu SORTED.")
    print(f"   SORTED rămâne metoda mai EFICIENTĂ!")
