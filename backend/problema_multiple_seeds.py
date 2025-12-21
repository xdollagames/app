#!/usr/bin/env python3
"""
PROBLEMA CRITICĂ: Multiple seeds → Multiple predicții diferite!
"""

from advanced_rng_library import create_rng, generate_numbers

# Extragerea din 2025-12-11
target = [6, 27, 9, 31, 4, 11]
target_sorted = sorted(target)

# Toate cele 3 seeds găsite pentru ACEEAȘI extragere
seeds_found = [
    626073,
    2116949,
    2692990
]

print("=" * 80)
print("🚨 PROBLEMA CRITICĂ: Multiple Seeds pentru Aceeași Extragere")
print("=" * 80)
print(f"\n🎯 Extragere din 2025-12-11: {target}")
print(f"🔢 Set sortat: {target_sorted}")
print(f"\n✅ Am găsit {len(seeds_found)} seeds diferite care generează acest set!")

print("\n" + "=" * 80)
print("📊 VERIFICARE: Fiecare seed generează setul corect?")
print("=" * 80)

for i, seed in enumerate(seeds_found, 1):
    rng = create_rng('lcg_minstd', seed)
    generated = generate_numbers(rng, 6, 1, 40)
    match = sorted(generated) == target_sorted
    print(f"\n{i}. Seed {seed:,}:")
    print(f"   Generează: {generated}")
    print(f"   Sortat:    {sorted(generated)}")
    print(f"   Match:     {'✅' if match else '❌'}")

print("\n" + "=" * 80)
print("🔮 PREDICȚIA URMĂTOARE: Care seed să folosim???")
print("=" * 80)

predictions = []

for i, seed in enumerate(seeds_found, 1):
    # Simulăm: folosim seed-ul ca și cum am "prezis" extragerea trecută
    # Acum vrem să prezicem URMĂTOAREA extragere
    rng = create_rng('lcg_minstd', seed)
    
    # Generăm extragerea "trecută" (cea pe care am găsit-o)
    past = generate_numbers(rng, 6, 1, 40)
    
    # Generăm URMĂTOAREA extragere (predicția!)
    next_prediction = generate_numbers(rng, 6, 1, 40)
    
    predictions.append(next_prediction)
    
    print(f"\n{i}. Cu Seed {seed:,}:")
    print(f"   Extragere trecută (match): {past} → {sorted(past)}")
    print(f"   🔮 PREDICȚIE următoare:   {next_prediction} → {sorted(next_prediction)}")

print("\n" + "=" * 80)
print("❓ PROBLEMA: Care predicție e corectă?")
print("=" * 80)

# Verificăm dacă predicțiile sunt diferite
unique_predictions = set(tuple(sorted(p)) for p in predictions)

if len(unique_predictions) > 1:
    print(f"\n🚨 PROBLEMĂ CRITICĂ: Avem {len(unique_predictions)} PREDICȚII DIFERITE!")
    print(f"\n   Toate cele 3 seeds generează setul corect pentru 2025-12-11,")
    print(f"   dar fiecare prezice ceva DIFERIT pentru următoarea extragere!")
    print(f"\n   ❓ Care seed e \"adevăratul\"? Nu avem cum să știm!")
elif len(unique_predictions) == 1:
    print(f"\n✅ Norocoși! Toate predicțiile sunt identice (set).")
    print(f"   Dar aceasta e o coincidență rară!")

print("\n" + "=" * 80)
print("🎓 CONCLUZIE:")
print("=" * 80)
print(f"\n✅ DA, ai perfect dreptate!")
print(f"\nCu SORTED găsim multiple seeds pentru aceeași extragere,")
print(f"și fiecare seed dă o PREDICȚIE DIFERITĂ pentru viitor!")
print(f"\n🚨 Asta DISTRUGE 100% acuratețea predicției!")
print(f"\n💡 PROBLEMA FUNDAMENTALĂ:")
print(f"   Reverse-engineering RNG funcționează pentru:")
print(f"   ✅ Loterii PSEUDO-ALEATOARE (software/online) - ordinea contează")
print(f"   ❌ Loterii FIZICE (bile reale) - ordinea e ALEATOARE")
print(f"\n🎯 Pentru loterii fizice, metoda RNG NU este viabilă!")
print(f"   Ar trebui folosite modele STATISTICE sau ML, nu RNG reverse-engineering!")

# Calculăm câte combinații diferite putem avea
print("\n" + "=" * 80)
print("📊 STATISTICI SUPLIMENTARE:")
print("=" * 80)
print(f"\nPentru setul {target_sorted}:")
print(f"   - Găsit {len(seeds_found)} seeds în 4,000,000 testați")
print(f"   - Fiecare seed generează o ordine diferită")
print(f"   - Fiecare seed → predicție diferită")
print(f"\nDacă am testa TOATE seeds-urile posibile (2^31-1),")
print(f"am putea găsi sute sau mii de seeds pentru același set!")
print(f"Asta înseamnă sute/mii de predicții DIFERITE pentru aceeași extragere trecută!")
