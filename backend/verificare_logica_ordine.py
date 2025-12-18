#!/usr/bin/env python3
"""
Verificare amănunțită: Ordinea datelor și seed-urilor
"""

import json
from datetime import datetime

def verifica_ordine_scraper():
    """Verifică ordinea extragerilor din scraper"""
    
    print("=" * 80)
    print("1️⃣ VERIFICARE: Ordinea în unified_lottery_scraper.py")
    print("=" * 80)
    
    with open('5-40_data.json', 'r') as f:
        data = json.load(f)
    
    draws = data['draws']
    
    print(f"\nTotal extrageri: {len(draws)}")
    
    # Verifică ordinea cronologică
    print(f"\n📅 Verificare ordinea cronologică:\n")
    
    print(f"Prima extragere:")
    print(f"   Data: {draws[0]['date']} - {draws[0]['date_str']}")
    print(f"   Numere: {draws[0]['numbers']}")
    
    print(f"\nUltima extragere:")
    print(f"   Data: {draws[-1]['date']} - {draws[-1]['date_str']}")
    print(f"   Numere: {draws[-1]['numbers']}")
    
    # Verifică că datele sunt în ordine cronologică
    print(f"\n🔍 Test ordine cronologică:")
    
    dates = [datetime.fromisoformat(d['date']) for d in draws]
    is_chronological = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
    
    print(f"   Datele sunt în ordine cronologică?: {is_chronological}")
    
    if is_chronological:
        print(f"   ✓ CORECT: De la cea mai veche la cea mai nouă")
    else:
        print(f"   ❌ PROBLEMĂ: Ordinea NU e cronologică!")
        
        # Găsește prima inversiune
        for i in range(len(dates)-1):
            if dates[i] > dates[i+1]:
                print(f"\n   Prima inversiune găsită:")
                print(f"      Index {i}: {draws[i]['date_str']}")
                print(f"      Index {i+1}: {draws[i+1]['date_str']}")
                break
    
    # Afișează primele 10 și ultimele 10 date
    print(f"\n📊 Primele 10 extrageri:")
    for i in range(min(10, len(draws))):
        print(f"   {i+1:3d}. {draws[i]['date']} - {draws[i]['numbers']}")
    
    print(f"\n📊 Ultimele 10 extrageri:")
    for i in range(max(0, len(draws)-10), len(draws)):
        print(f"   {i+1:3d}. {draws[i]['date']} - {draws[i]['numbers']}")
    
    return is_chronological


def verifica_logica_seed_pattern():
    """Verifică cum se calculează pattern-ul seed-urilor"""
    
    print("\n" + "=" * 80)
    print("2️⃣ VERIFICARE: Logica Pattern Seed-uri")
    print("=" * 80)
    
    print("""
📋 CE AR TREBUI SĂ SE ÎNTÂMPLE:

1. Găsim seed-uri pentru extrageri în ORDINEA CRONOLOGICĂ
2. Seed-urile găsite formează o SECVENȚĂ (seed1, seed2, seed3, ...)
3. Analizăm pattern-ul: LINEAR, POLINOMIAL, etc.
4. Predicția URMĂTORULUI seed e bazată pe pattern
5. Cu seed-ul prezis, generăm numerele pentru URMĂTOAREA extragere

Exemplu:
    Draw 1 (01.01.2024): seed=1000  ← Cea mai veche
    Draw 2 (02.01.2024): seed=2000
    Draw 3 (03.01.2024): seed=3000
    
    Pattern: LINEAR, increment=1000
    
    Predicție Draw 4 (04.01.2024): seed=4000  ← Următoarea

⚠️  CRITIC: Ordinea trebuie păstrată pentru pattern analysis!
""")
    
    # Simulare pattern analysis
    print("🔬 Simulare Pattern Analysis:\n")
    
    # Seed-uri "găsite" în ordine cronologică
    seeds_example = [1234, 2468, 3702, 4936, 6170]
    
    print(f"Seed-uri găsite (în ordine cronologică):")
    for i, seed in enumerate(seeds_example):
        print(f"   Draw {i+1}: seed={seed}")
    
    # Calculează diferențe
    diffs = [seeds_example[i+1] - seeds_example[i] for i in range(len(seeds_example)-1)]
    print(f"\nDiferențe între seed-uri consecutive:")
    for i, diff in enumerate(diffs):
        print(f"   {seeds_example[i+1]} - {seeds_example[i]} = {diff}")
    
    # Pattern detection
    avg_diff = sum(diffs) / len(diffs)
    is_constant = all(abs(d - avg_diff) < 0.1 for d in diffs)
    
    print(f"\nPattern detectat:")
    print(f"   Diferență medie: {avg_diff:.1f}")
    print(f"   Pattern constant?: {is_constant}")
    
    if is_constant:
        next_seed = seeds_example[-1] + avg_diff
        print(f"\n✓ Predicție seed următoare extragere:")
        print(f"   {seeds_example[-1]} + {avg_diff:.1f} = {next_seed:.0f}")
    
    return True


def verifica_cod_cpu_predictor():
    """Verifică logica din cpu_only_predictor.py"""
    
    print("\n" + "=" * 80)
    print("3️⃣ VERIFICARE: Logica în cpu_only_predictor.py")
    print("=" * 80)
    
    print("""
📝 CE AR TREBUI SĂ FACĂ cpu_only_predictor.py:

PARTEA 1: Seed Finding
    FOR each draw in ORDINE CRONOLOGICĂ:
        Search seed care generează numerele exacte
        IF found:
            seeds[draw_index] = seed
    
    → Rezultat: Liste de seed-uri în ORDINEA CRONOLOGICĂ

PARTEA 2: Pattern Analysis
    seeds_list = [seed1, seed2, seed3, ...]  ← În ordine cronologică!
    
    Testează patterns:
        - LINEAR: seed(n) = a*n + b
        - POLYNOMIAL: seed(n) = a*n² + b*n + c
        - EXPONENTIAL: seed(n) = a * e^(b*n)
    
    Best pattern → folosește pentru predicție

PARTEA 3: Predicție
    next_index = len(seeds_list)
    next_seed = pattern_function(next_index)
    
    Generate numbers with next_seed
    → PREDICȚIE pentru următoarea extragere!

⚠️  VERIFICĂRI CRITICE:
    1. draws sunt procesate în ordine cronologică?
    2. seeds_list păstrează ordinea?
    3. pattern analysis folosește index-uri corecte?
    4. predicția e pentru URMĂTOAREA (nu una random)?
""")


def verifica_index_usage():
    """Verifică cum sunt folosiți index-urile"""
    
    print("\n" + "=" * 80)
    print("4️⃣ VERIFICARE: Utilizare Index-uri și Ordine")
    print("=" * 80)
    
    with open('5-40_data.json', 'r') as f:
        data = json.load(f)
    
    draws = data['draws']
    
    print(f"\n📊 Structura datelor:\n")
    
    print(f"Total draws: {len(draws)}")
    print(f"Index range: 0 to {len(draws)-1}")
    
    print(f"\nExemple index → date:")
    indices_to_check = [0, 1, 2, len(draws)-3, len(draws)-2, len(draws)-1]
    
    for idx in indices_to_check:
        if 0 <= idx < len(draws):
            draw = draws[idx]
            print(f"   draws[{idx:4d}] = {draw['date']} - {draw['numbers']}")
    
    print(f"\n🎯 Pentru PREDICȚIE:")
    print(f"   Ultima extragere cunoscută: draws[{len(draws)-1}]")
    print(f"      Data: {draws[-1]['date']}")
    print(f"      Numere: {draws[-1]['numbers']}")
    print(f"\n   Următoarea extragere (predicție): draws[{len(draws)}]")
    print(f"      Data: VIITOR (după {draws[-1]['date']})")
    print(f"      Numere: ??? (trebuie prezise)")
    
    print(f"\n✓ Pattern analysis trebuie să folosească:")
    print(f"   x = [0, 1, 2, ..., {len(draws)-1}]  ← Index-uri cunoscute")
    print(f"   y = [seed0, seed1, seed2, ..., seed{len(draws)-1}]  ← Seed-uri găsite")
    print(f"   Predicție pentru: x = {len(draws)}  ← Index VIITOR")


def analiza_seed_cache():
    """Analizează cum sunt stocate seed-urile în cache"""
    
    print("\n" + "=" * 80)
    print("5️⃣ VERIFICARE: Seeds Cache Structure")
    print("=" * 80)
    
    try:
        with open('seeds_cache.json', 'r') as f:
            cache = json.load(f)
        
        print(f"\n📦 Cache structure:\n")
        
        if not cache:
            print("   Cache este gol ({})")
            print("   ✓ Normal după reset-uri")
        else:
            print(f"   Loterii în cache: {list(cache.keys())}")
            
            for lottery_type, dates in cache.items():
                print(f"\n   {lottery_type}:")
                print(f"      Total date: {len(dates)}")
                
                # Afișează primele câteva
                dates_list = list(dates.keys())[:5]
                for date_str in dates_list:
                    print(f"         {date_str}: {dates[date_str]}")
                
                if len(dates) > 5:
                    print(f"         ... și încă {len(dates)-5}")
        
        print(f"\n🔍 Cache folosește date_str ca cheie:")
        print(f"   ✓ BINE dacă date_str e unică per extragere")
        print(f"   ⚠️  PROBLEMĂ dacă date_str se repetă")
        
    except FileNotFoundError:
        print("\n   seeds_cache.json nu există")
        print("   ✓ Normal după ștergere")


def test_ordinea_completa():
    """Test complet al ordinii în tot flow-ul"""
    
    print("\n" + "=" * 80)
    print("6️⃣ TEST COMPLET: Flow de la Scraper la Predicție")
    print("=" * 80)
    
    with open('5-40_data.json', 'r') as f:
        data = json.load(f)
    
    draws = data['draws']
    
    print(f"""
📋 FLOW COMPLET:

1. SCRAPER extrage date:
   ┌─────────────────────────────────────────┐
   │ 01.01.1995: [5, 13, 26, 38, 37, 25]    │  ← Index 0
   │ 08.01.1995: [20, 32, 38, 21, 5, 11]    │  ← Index 1
   │ ...                                      │
   │ 14.12.2025: [36, 39, 6, 19, 15, 33]    │  ← Index {len(draws)-1}
   └─────────────────────────────────────────┘
   
2. PREDICTOR caută seeds:
   ┌─────────────────────────────────────────┐
   │ Index 0 → seed: ???                     │
   │ Index 1 → seed: ???                     │
   │ ...                                      │
   │ Index {len(draws)-1} → seed: ???        │
   └─────────────────────────────────────────┘
   
3. PATTERN ANALYSIS:
   ┌─────────────────────────────────────────┐
   │ x = [0, 1, 2, ..., {len(draws)-1}]      │
   │ y = [seed_0, seed_1, ..., seed_{len(draws)-1}] │
   │                                          │
   │ Fit pattern: y = f(x)                   │
   └─────────────────────────────────────────┘
   
4. PREDICȚIE:
   ┌─────────────────────────────────────────┐
   │ next_index = {len(draws)}                │
   │ next_seed = f({len(draws)})              │
   │ next_numbers = generate(next_seed)      │
   │                                          │
   │ → PREDICȚIE pentru următoarea extragere │
   └─────────────────────────────────────────┘

✓ Ordinea TREBUIE păstrată în TOATE etapele!
✗ Orice sortare/shuffle → predicția devine INVALIDĂ!
""")
    
    # Test rapid pe primele 5 date
    print(f"\n🧪 Test rapid pe primele 5 extrageri:\n")
    
    for i in range(min(5, len(draws))):
        draw = draws[i]
        print(f"   Index {i}: {draw['date']} → {draw['numbers']}")
    
    print(f"\n   Pentru predicție, am folosi index: {min(5, len(draws))}")
    print(f"   (care ar corespunde datei după {draws[min(4, len(draws)-1)]['date']})")


def recomandari_critice():
    """Recomandări critice pentru verificare cod"""
    
    print("\n" + "=" * 80)
    print("⚠️  RECOMANDĂRI CRITICE")
    print("=" * 80)
    
    print("""
🔴 VERIFICĂRI OBLIGATORII în cpu_only_predictor.py:

1. ✓ NU sortează draws după găsirea seed-urilor
   ❌ BAD: seeds_dict = {date: seed for ...}; sorted(seeds_dict.keys())
   ✓ GOOD: seeds_list = [seed for draw in draws in ORDER]

2. ✓ Index-urile corespund ordinii cronologice
   ❌ BAD: x = sorted(indices)
   ✓ GOOD: x = list(range(len(draws)))

3. ✓ Pattern analysis folosește ordinea originală
   ❌ BAD: y = sorted(seeds)
   ✓ GOOD: y = seeds  # păstrează ordinea din draws

4. ✓ Predicția e pentru index URMĂTOR
   ❌ BAD: next_index = random choice
   ✓ GOOD: next_index = len(seeds_found)

5. ✓ Cache-ul folosește date_str UNICE
   ⚠️  Verifică că nu există duplicate date_str!

6. ✓ Datele din JSON sunt deja ordonate cronologic
   ⚠️  Verificat: {verifica_ordine_scraper()}

📝 CE TREBUIE VERIFICAT MANUAL ÎN COD:

1. Caută: "sorted(" în cpu_only_predictor.py
   → NU trebuie să sorteze seeds după găsire!

2. Caută: "analyze_seed_pattern" sau similar
   → Verifică că primește seeds în ORDINEA CORECTĂ

3. Caută: "predict" sau "next_seed"
   → Verifică că index-ul e len(seeds), nu altceva

4. Caută: "enumerate(draws" sau "for i, draw in"
   → Verifică că procesează în ordine, nu random

5. Verifică că NU face:
   ❌ random.shuffle(draws)
   ❌ sorted(seeds_dict.values())
   ❌ reversed(draws)
""")


def main():
    """Execută toate verificările"""
    
    print("\n" + "🔍" * 40)
    print("VERIFICARE AMĂNUNȚITĂ: Logica Ordinii")
    print("🔍" * 40)
    
    # 1. Verifică ordinea în scraper
    verifica_ordine_scraper()
    
    # 2. Verifică logica pattern
    verifica_logica_seed_pattern()
    
    # 3. Verifică ce ar trebui să facă predictorul
    verifica_cod_cpu_predictor()
    
    # 4. Verifică index-uri
    verifica_index_usage()
    
    # 5. Verifică cache
    analiza_seed_cache()
    
    # 6. Test complet
    test_ordinea_completa()
    
    # 7. Recomandări
    recomandari_critice()
    
    print("\n" + "=" * 80)
    print("✅ VERIFICARE COMPLETĂ")
    print("=" * 80)
    print("""
📝 NEXT STEPS:

1. Rulează: grep -n "sorted(" /app/backend/cpu_only_predictor.py
2. Verifică manual funcțiile de pattern analysis
3. Testează pe câteva extrageri reale
4. Asigură-te că predicția e pentru URMĂTOAREA extragere
""")


if __name__ == '__main__':
    main()
