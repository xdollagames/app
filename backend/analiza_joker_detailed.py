#!/usr/bin/env python3
"""
Analiză detaliată Joker - înțelegem logica corectă de generare
"""

import json
from collections import Counter

def analiza_completa_joker():
    """Analiză comprehensivă date Joker"""
    
    with open('joker_data.json', 'r') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("🎰 ANALIZĂ DETALIATĂ JOKER")
    print("=" * 80)
    
    print(f"\nTotal extrageri: {data['total_draws']}")
    print(f"Config: {data['config']}")
    
    # Analiză structură
    print("\n" + "=" * 80)
    print("1️⃣ ANALIZA STRUCTURII")
    print("=" * 80)
    
    for i, draw in enumerate(data['draws'][:10], 1):
        print(f"\n{i}. {draw['date_str']}")
        print(f"   Numere complete: {draw['numbers']}")
        
        if len(draw['numbers']) == 6:
            primele_5 = draw['numbers'][:5]
            joker = draw['numbers'][5]
            
            print(f"   Primele 5:       {primele_5}")
            print(f"   Joker (nr 6):    {joker}")
            
            # Verificări
            toate_in_range_45 = all(1 <= n <= 45 for n in primele_5)
            joker_in_range_20 = 1 <= joker <= 20
            joker_duplicate = joker in primele_5
            
            print(f"   Primele 5 în 1-45?: {toate_in_range_45} {'✓' if toate_in_range_45 else '❌'}")
            print(f"   Joker în 1-20?:     {joker_in_range_20} {'✓' if joker_in_range_20 else '❌'}")
            print(f"   Joker duplicat?:    {joker_duplicate} {'❌ PROBLEMA!' if joker_duplicate else '✓'}")

def analiza_duplicate_joker():
    """Verifică dacă Joker-ul poate fi duplicat cu primele 5"""
    
    with open('joker_data.json', 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("2️⃣ ANALIZA DUPLICATE JOKER")
    print("=" * 80)
    
    duplicate_count = 0
    non_duplicate_count = 0
    duplicate_examples = []
    
    for draw in data['draws']:
        if len(draw['numbers']) == 6:
            primele_5 = draw['numbers'][:5]
            joker = draw['numbers'][5]
            
            if joker in primele_5:
                duplicate_count += 1
                if len(duplicate_examples) < 5:
                    duplicate_examples.append({
                        'date': draw['date_str'],
                        'numbers': draw['numbers'],
                        'primele_5': primele_5,
                        'joker': joker
                    })
            else:
                non_duplicate_count += 1
    
    total = duplicate_count + non_duplicate_count
    
    print(f"\n📊 Statistici Duplicate:")
    print(f"   Total extrageri analizate: {total}")
    print(f"   Joker DUPLICAT cu primele 5: {duplicate_count} ({duplicate_count/total*100:.1f}%)")
    print(f"   Joker UNIC (nu e în primele 5): {non_duplicate_count} ({non_duplicate_count/total*100:.1f}%)")
    
    if duplicate_examples:
        print(f"\n⚠️  EXEMPLE DUPLICATE găsite:")
        for ex in duplicate_examples:
            print(f"   {ex['date']}")
            print(f"      Toate: {ex['numbers']}")
            print(f"      Primele 5: {ex['primele_5']}")
            print(f"      Joker: {ex['joker']} ← DUPLICAT!")
    
    print(f"\n{'✓' if duplicate_count == 0 else '⚠️'} ", end="")
    if duplicate_count == 0:
        print("CONCLUZIE: Joker-ul este ÎNTOTDEAUNA unic (nu se repetă cu primele 5)")
    else:
        print(f"CONCLUZIE: Joker-ul POATE fi duplicat ({duplicate_count} cazuri găsite)")

def analiza_range_joker():
    """Analizează range-ul efectiv al Joker-ului"""
    
    with open('joker_data.json', 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("3️⃣ ANALIZA RANGE JOKER")
    print("=" * 80)
    
    valori_joker = []
    valori_primele_5 = []
    
    for draw in data['draws']:
        if len(draw['numbers']) == 6:
            primele_5 = draw['numbers'][:5]
            joker = draw['numbers'][5]
            
            valori_joker.append(joker)
            valori_primele_5.extend(primele_5)
    
    # Statistici Joker
    min_joker = min(valori_joker)
    max_joker = max(valori_joker)
    valori_unice_joker = len(set(valori_joker))
    
    print(f"\n📊 Statistici Joker (nr 6):")
    print(f"   Range teoretic: 1-20")
    print(f"   Range efectiv:  {min_joker}-{max_joker}")
    print(f"   Valori unice:   {valori_unice_joker}/20")
    
    # Frecvență Joker
    freq_joker = Counter(valori_joker)
    print(f"\n   Top 5 cele mai frecvente:")
    for val, count in freq_joker.most_common(5):
        print(f"      {val:2d}: {count:3d} apariții ({count/len(valori_joker)*100:.1f}%)")
    
    # Statistici primele 5
    min_5 = min(valori_primele_5)
    max_5 = max(valori_primele_5)
    valori_unice_5 = len(set(valori_primele_5))
    
    print(f"\n📊 Statistici Primele 5:")
    print(f"   Range teoretic: 1-45")
    print(f"   Range efectiv:  {min_5}-{max_5}")
    print(f"   Valori unice:   {valori_unice_5}/45")
    
    # Verifică overlap
    overlap_range = set(range(1, 21)) & set(valori_primele_5)
    print(f"\n🔍 Overlap între range-uri:")
    print(f"   Range Joker (1-20) găsit în primele 5?: DA, {len(overlap_range)} valori")
    print(f"   Valori din 1-20 în primele 5: {sorted(overlap_range)[:10]}...")

def analiza_secventa_generare():
    """Încearcă să determine dacă e secvență consecutivă sau separate"""
    
    with open('joker_data.json', 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 80)
    print("4️⃣ ANALIZA SECVENȚĂ vs SEPARATE")
    print("=" * 80)
    
    print("\n📝 Ipoteze de Testare:")
    
    print("\n🔹 IPOTEZA A: Secvență Consecutivă")
    print("   RNG generează 6 numere consecutive în range 1-45:")
    print("   - Primele 5 unique → Partea 1 (toate în 1-45)")
    print("   - Nr 6 unique care e în 1-20 → Joker")
    print("   - Dacă nr 6 > 20, continuă până găsește unul în 1-20")
    
    print("\n🔹 IPOTEZA B: Două Secvențe Separate")
    print("   RNG 1: Generează 5 numere în 1-45")
    print("   RNG 2: Generează 1 număr în 1-20 (poate același RNG, dar range diferit)")
    
    print("\n🔹 IPOTEZA C: Secvență cu Filtrare")
    print("   RNG generează numere consecutive:")
    print("   - Colectează 5 numere unique din 1-45")
    print("   - Apoi colectează 1 număr din 1-20 care NU e duplicat")
    
    # Teste pentru fiecare ipoteză
    print("\n" + "-" * 80)
    print("TESTE:")
    
    # Test: Verifică dacă Joker > 20 există
    joker_over_20 = 0
    joker_in_range = 0
    
    for draw in data['draws']:
        if len(draw['numbers']) == 6:
            joker = draw['numbers'][5]
            if joker > 20:
                joker_over_20 += 1
            else:
                joker_in_range += 1
    
    print(f"\n✓ Test 1: Joker peste 20?")
    print(f"   Joker în 1-20:  {joker_in_range} ({joker_in_range/(joker_in_range+joker_over_20)*100:.1f}%)")
    print(f"   Joker > 20:     {joker_over_20} ({joker_over_20/(joker_in_range+joker_over_20)*100:.1f}%)")
    
    if joker_over_20 == 0:
        print(f"   → Suportă IPOTEZA B sau C (range restricționat de la început)")
    else:
        print(f"   → Suportă IPOTEZA A (filtrare post-generare)")

def propunere_implementare():
    """Propune implementarea corectă bazată pe analiză"""
    
    print("\n" + "=" * 80)
    print("5️⃣ PROPUNERE IMPLEMENTARE")
    print("=" * 80)
    
    print("\n📋 Bazat pe analiză, recomandarea este:")
    
    print("\n🎯 ABORDARE RECOMANDATĂ: Ipoteza C (Secvență cu Filtrare)")
    
    print("\n```python")
    print("def generate_joker_sequence(rng):")
    print("    \"\"\"")
    print("    Generează 6 numere pentru Joker:")
    print("    - Primele 5: unique din 1-45")
    print("    - Nr 6 (Joker): unique din 1-20, diferit de primele 5")
    print("    \"\"\"")
    print("    numbers = []")
    print("    seen = set()")
    print("    ")
    print("    # Partea 1: Generează 5 numere din 1-45")
    print("    attempts = 0")
    print("    while len(numbers) < 5 and attempts < 500:")
    print("        num = 1 + (rng.next() % 45)  # Range 1-45")
    print("        if num not in seen:")
    print("            numbers.append(num)")
    print("            seen.add(num)")
    print("        attempts += 1")
    print("    ")
    print("    # Partea 2: Generează Joker din 1-20 (diferit de primele 5)")
    print("    attempts = 0")
    print("    while len(numbers) < 6 and attempts < 500:")
    print("        num = 1 + (rng.next() % 20)  # Range 1-20")
    print("        if num not in seen:")
    print("            numbers.append(num)")
    print("            seen.add(num)")
    print("            break")
    print("        attempts += 1")
    print("    ")
    print("    return numbers")
    print("```")
    
    print("\n⚠️  ALTERNATIVĂ: Testare Multiplă")
    print("   Dacă nu știm exact metoda, putem testa TOATE ipotezele:")
    print("   1. Testează Ipoteza A (consecutiv, filtrare)")
    print("   2. Testează Ipoteza B (două secvențe separate)")
    print("   3. Testează Ipoteza C (consecutiv cu skip)")
    print("   → Vezi care găsește seed-uri!")

def main():
    """Rulează toate analizele"""
    
    print("\n" + "🎰" * 40)
    print("INVESTIGAȚIE JOKER: Determinare Logică Corectă")
    print("🎰" * 40 + "\n")
    
    # Analiză 1: Structură
    analiza_completa_joker()
    
    # Analiză 2: Duplicate
    analiza_duplicate_joker()
    
    # Analiză 3: Range
    analiza_range_joker()
    
    # Analiză 4: Secvență
    analiza_secventa_generare()
    
    # Propunere
    propunere_implementare()
    
    print("\n" + "=" * 80)
    print("✅ ANALIZĂ COMPLETĂ")
    print("=" * 80)
    print("\n📝 Următorii pași:")
    print("   1. Implementează funcția generate_joker_sequence()")
    print("   2. Testează cu date reale")
    print("   3. Compară rezultate între ipoteze")
    print("   4. Validează seed-urile găsite")

if __name__ == '__main__':
    main()
