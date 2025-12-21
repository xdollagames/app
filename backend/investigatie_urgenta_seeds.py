#!/usr/bin/env python3
"""
Investigație CRITICĂ: Sunt numerele în JSON sortate sau în ordine de extragere?
"""

import json

def analiza_daca_sunt_sortate():
    """Verifică dacă numerele din JSON sunt deja sortate"""
    
    print("=" * 80)
    print("🚨 INVESTIGAȚIE CRITICĂ: Ordinea Numerelor în Date")
    print("=" * 80)
    
    with open('5-40_data.json', 'r') as f:
        data = json.load(f)
    
    draws = data['draws']
    
    print(f"\nTotal extrageri: {len(draws)}")
    
    # Verifică câte sunt deja sortate
    sortate_count = 0
    nesortate_count = 0
    
    print(f"\n📊 Analiza primelor 30 extrageri:\n")
    
    for i, draw in enumerate(draws[:30]):
        numbers = draw['numbers']
        sorted_nums = sorted(numbers)
        is_sorted = (numbers == sorted_nums)
        
        if is_sorted:
            sortate_count += 1
            status = "✓ SORTATE"
        else:
            nesortate_count += 1
            status = "✗ Nesortate"
        
        print(f"{i+1:3d}. {draw['date_str'][:20]:20s} {numbers} {status}")
    
    print(f"\n{'='*80}")
    print(f"📈 STATISTICI PE PRIMELE 30:")
    print(f"{'='*80}")
    print(f"   Deja sortate:    {sortate_count}/30 ({sortate_count/30*100:.1f}%)")
    print(f"   NU sortate:      {nesortate_count}/30 ({nesortate_count/30*100:.1f}%)")
    
    # Acum pe TOATE
    print(f"\n{'='*80}")
    print(f"📈 STATISTICI PE TOATE {len(draws)} EXTRAGERILE:")
    print(f"{'='*80}")
    
    sortate_total = 0
    nesortate_total = 0
    
    for draw in draws:
        numbers = draw['numbers']
        if numbers == sorted(numbers):
            sortate_total += 1
        else:
            nesortate_total += 1
    
    print(f"   Deja sortate:    {sortate_total}/{len(draws)} ({sortate_total/len(draws)*100:.1f}%)")
    print(f"   NU sortate:      {nesortate_total}/{len(draws)} ({nesortate_total/len(draws)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"🎯 CONCLUZIE:")
    print(f"{'='*80}")
    
    if sortate_total / len(draws) > 0.95:
        print(f"""
❌ PROBLEMĂ CRITICĂ IDENTIFICATĂ!

{sortate_total/len(draws)*100:.1f}% din extrageri sunt deja SORTATE în JSON!

Asta înseamnă:
    1. Site-ul noroc-chior.ro afișează numerele SORTATE
    2. Scraper-ul salvează numerele cum le vede pe site (sortate)
    3. Nu există "ordinea de extragere" în date!
    4. RNG analysis NU poate funcționa (nu avem ordinea reală!)

→ De aceea după fix, nu mai găsește seed-uri!

ÎNAINTE (cu comparație sortată):
    • Compara sorted(generated) cu sorted(target)
    • Ambele erau sortate → găsea "coincidențe"
    • False positives, dar părea să funcționeze

DUPĂ (cu comparație exactă):
    • Compară generated cu target (ambele sortate în date)
    • NU găsește pentru că RNG-urile generează în ordine DIFERITĂ
    • Nicio coincidență → corect, dar nefuncțional!

SOLUȚIE: Trebuie să comparăm SET-uri, nu liste ordonate!
""")
    elif sortate_total / len(draws) < 0.05:
        print(f"""
✓ EXCELENT! Doar {sortate_total/len(draws)*100:.1f}% sunt sortate!

Datele conțin ORDINEA REALĂ de extragere!

Dacă nu găsește seed-uri:
    → RNG-urile testate nu sunt cele folosite de loterie
    → NORMAL pentru loterie FIZICĂ!
""")
    else:
        print(f"""
⚠️  REZULTAT MIXT: {sortate_total/len(draws)*100:.1f}% sunt sortate

Trebuie investigat mai detaliat de ce unele sunt sortate și altele nu.
""")

def verifica_ordine_pe_site():
    """Sugestii pentru verificare site"""
    
    print(f"\n{'='*80}")
    print(f"🌐 VERIFICARE NECESARĂ: Site-ul noroc-chior.ro")
    print(f"{'='*80}")
    
    print("""
Pentru a determina dacă numerele sunt în ordine de extragere:

1. Accesează: http://noroc-chior.ro/Loto/5-din-40/arhiva-rezultate.php

2. Verifică o extragere recentă:
    → Sunt numerele afișate în ordine CRESCĂTOARE?
    → Sau sunt în ordinea de EXTRAGERE (random)?

3. Compară cu transmisia LIVE (dacă există înregistrări):
    → Care a fost ordinea REALĂ de extragere?
    → Corespunde cu ce e pe site?

4. Verifică documentația site-ului:
    → Specifică dacă ordinea e păstrată?
    → Sau sunt automat sortate?

Dacă site-ul afișează SORTATE:
    → Scraper-ul nu poate obține ordinea reală
    → RNG analysis devine IMPOSIBIL
    → Trebuie pivota spre statistici!
""")

def propunere_solutie():
    """Propune soluția bazată pe situație"""
    
    print(f"\n{'='*80}")
    print(f"✅ SOLUȚII POSIBILE")
    print(f"{'='*80}")
    
    print("""
SOLUȚIA 1: Revenim la comparație cu SET-uri (dacă datele sunt sortate)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dacă datele din JSON sunt SORTATE (nu ordinea reală):

```python
# În loc de:
if generated == target:  # Compară ordinea exactă

# Folosește:
if set(generated) == set(target):  # Compară SET-uri (ignore ordinea)
```

PRO:
    ✓ Va găsi seed-uri (dacă RNG-ul e corect)
    ✓ Funcțional cu date sortate
    ✓ Elimină dependency de ordine

CONTRA:
    ❌ FALSE POSITIVES (multe seed-uri diferite cu aceleași numere)
    ❌ Nu știi care seed e "corect"
    ❌ Pattern analysis devine imprecisă

SOLUȚIA 2: Acceptă realitatea - Pivotează spre STATISTICI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pentru LOTERIE FIZICĂ cu date SORTATE:

    • RNG seed finding = IMPOSIBIL și INUTIL
    • Pattern-urile RNG sunt coincidențe
    • Pivotează spre STATISTICI și ML

Features utile:
    ✓ Frecvențe numere (all-time, recent, yearly)
    ✓ Hot/Cold numbers (ultimele X extrageri)
    ✓ Perechi frecvente, triplete
    ✓ Analiză paritate (pare/impare)
    ✓ Analiză sume, intervale
    ✓ ML pentru pattern-uri PROBABILISTICE
    ✓ Generare combinații "smart" bazate pe stats

SOLUȚIA 3: Găsește ordinea REALĂ de extragere (ideal dar dificil)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Surse posibile:
    • Înregistrări video ale transmisiunilor LIVE
    • Site-uri care păstrează ordinea originală
    • API-uri oficiale (dacă există)
    • OCR pe video-uri YouTube

Dificultate: ÎNALTĂ
    ⚠️  Procesare video complexă
    ⚠️  OCR pentru fiecare extragere
    ⚠️  30 ani de date = mii de video-uri
    ⚠️  Timp și resurse considerabile

""")

def test_cu_set_comparison():
    """Test: Ce se întâmplă dacă comparăm SET-uri"""
    
    print(f"\n{'='*80}")
    print(f"🧪 TEST: Comparație SET vs LISTĂ")
    print(f"{'='*80}")
    
    from advanced_rng_library import create_rng, generate_numbers
    
    # Date reale sortate (din JSON)
    target_sorted = [5, 13, 25, 26, 37, 38]
    
    print(f"\nTarget (sortat în JSON): {target_sorted}\n")
    
    # Generăm cu RNG
    found_count = 0
    seeds_found = []
    
    print(f"Test pe 100,000 seeds cu xorshift32:")
    for seed in range(0, 100000, 100):
        rng = create_rng('xorshift32', seed)
        generated = generate_numbers(rng, 6, 1, 40)
        
        # Test SET comparison
        if set(generated) == set(target_sorted):
            found_count += 1
            seeds_found.append((seed, generated))
            if found_count <= 5:
                print(f"   Seed {seed:6d}: {generated}")
    
    print(f"\n📊 Rezultate:")
    print(f"   Seeds găsite cu SET comparison: {found_count}")
    print(f"   {found_count} seed-uri diferite generează aceleași 6 numere!")
    
    if found_count > 0:
        print(f"\n⚠️  PROBLEMA: {found_count} FALSE POSITIVES!")
        print(f"   Toate aceste seed-uri par 'corecte'")
        print(f"   Dar genereză în ordini DIFERITE:")
        for seed, gen in seeds_found[:3]:
            print(f"      Seed {seed}: {gen}")

def main():
    """Execută investigația"""
    
    print("\n" + "🚨" * 40)
    print("INVESTIGAȚIE URGENTĂ: De ce nu mai găsește seed-uri?")
    print("🚨" * 40 + "\n")
    
    # Analiză dacă sunt sortate
    analiza_daca_sunt_sortate()
    
    # Verificare site
    verifica_ordine_pe_site()
    
    # Soluții
    propunere_solutie()
    
    # Test SET comparison
    test_cu_set_comparison()
    
    print(f"\n{'='*80}")
    print(f"📝 NEXT STEPS URGENT")
    print(f"{'='*80}")
    print("""
1. Verifică manual pe noroc-chior.ro dacă numerele sunt sortate
2. Dacă DA (sortate) → Revino la comparație SET
3. Dacă NU (ordine reală) → Investighează de ce nu găsește
4. Consideră pivot spre statistici (mai realistic pentru fizic)
""")

if __name__ == '__main__':
    main()
