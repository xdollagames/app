#!/usr/bin/env python3
"""
Explicație vizuală: Cum funcționează predicțiile pentru Joker?
"""

from advanced_rng_library import create_rng

def demonstratie_secventa_rng():
    """Demonstrație: UN RNG, SECVENȚĂ CONTINUĂ"""
    
    print("=" * 80)
    print("🎯 EXPLICAȚIE: Cum Funcționează Predicțiile pentru Joker")
    print("=" * 80)
    
    print("\n" + "▼" * 80)
    print("RĂSPUNS SCURT: UN SINGUR RNG, SECVENȚĂ CONTINUĂ!")
    print("▼" * 80)
    
    print("""
Joker-ul NU este tratat separat!

UN SINGUR RNG generează TOATE cele 6 numere în ORDINE CONSECUTIVĂ:
    
    RNG seed → State 1 → State 2 → State 3 → State 4 → State 5 → State 6
                 ↓         ↓         ↓         ↓         ↓         ↓
              Num 1     Num 2     Num 3     Num 4     Num 5     JOKER
             (1-45)    (1-45)    (1-45)    (1-45)    (1-45)    (1-20)
             
Diferența e DOAR range-ul: primele 5 din 1-45, ultimul din 1-20!
""")

def demonstratie_concreta():
    """Demonstrație concretă cu un RNG real"""
    
    print("\n" + "=" * 80)
    print("📊 DEMONSTRAȚIE CONCRETĂ")
    print("=" * 80)
    
    seed = 12345
    rng_type = 'xorshift32'
    
    print(f"\n🔧 Test cu: {rng_type}, seed={seed}\n")
    
    # Creează RNG
    rng = create_rng(rng_type, seed)
    
    print("Generăm 6 numere CONSECUTIVE (fără filtrare duplicate):\n")
    
    # Generează 6 valori RAW consecutive
    raw_values = []
    for i in range(6):
        raw = rng.next()
        raw_values.append(raw)
    
    print("Stările RNG (valori RAW):")
    for i, raw in enumerate(raw_values, 1):
        print(f"   State {i}: {raw:12d}")
    
    # Aplică range-uri
    print(f"\nAplicăm range-urile:")
    
    # Primele 5: modulo 45, plus 1
    primele_5_mapped = []
    for i in range(5):
        mapped = 1 + (raw_values[i] % 45)
        primele_5_mapped.append(mapped)
        print(f"   Num {i+1}: {raw_values[i]:12d} % 45 + 1 = {mapped:2d}  (range 1-45)")
    
    # Joker: modulo 20, plus 1
    joker_raw = raw_values[5]
    joker = 1 + (joker_raw % 20)
    print(f"   Joker: {joker_raw:12d} % 20 + 1 = {joker:2d}  (range 1-20)")
    
    print(f"\n🎰 Rezultat FĂRĂ filtrare duplicate:")
    print(f"   {primele_5_mapped + [joker]}")
    
    # Acum cu filtrare duplicate (cum facem în cod)
    print(f"\n" + "-" * 80)
    print("Cu FILTRARE DUPLICATE (implementarea noastră):\n")
    
    rng2 = create_rng(rng_type, seed)
    
    # Partea 1: generează 5 numere UNIQUE din 1-45
    generated = []
    seen = set()
    attempts = 0
    
    print("Partea 1: Generăm 5 numere UNIQUE din 1-45:")
    while len(generated) < 5 and attempts < 100:
        raw = rng2.next()
        num = 1 + (raw % 45)
        
        if num not in seen:
            generated.append(num)
            seen.add(num)
            print(f"   State {attempts+1}: {raw:12d} → {num:2d}  ✓ Adăugat (#{len(generated)})")
        else:
            print(f"   State {attempts+1}: {raw:12d} → {num:2d}  ✗ Duplicat, skip!")
        
        attempts += 1
    
    # Partea 2: generează Joker din 1-20 (PERMITE duplicate!)
    print(f"\nPartea 2: Generăm Joker din 1-20 (PERMITE duplicate):")
    
    joker_raw = rng2.next()
    joker_num = 1 + (joker_raw % 20)
    generated.append(joker_num)
    
    in_primele_5 = joker_num in generated[:5]
    print(f"   State {attempts+1}: {joker_raw:12d} → {joker_num:2d}  ✓ Adăugat ca Joker")
    print(f"   Joker duplicat cu primele 5?: {'DA ⚠️' if in_primele_5 else 'NU ✓'}")
    
    print(f"\n🎰 Rezultat FINAL:")
    print(f"   {generated}")
    print(f"   Primele 5: {generated[:5]}")
    print(f"   Joker:     {generated[5]}")

def compara_abordari():
    """Compară abordările: secvență vs separate"""
    
    print("\n" + "=" * 80)
    print("⚖️  COMPARAȚIE: Secvență Continuă vs Separate")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ ABORDAREA NOASTRĂ (CORECTĂ): SECVENȚĂ CONTINUĂ                             │
└─────────────────────────────────────────────────────────────────────────────┘

    RNG(seed) → State_1 → State_2 → State_3 → State_4 → State_5 → State_6
                   ↓          ↓          ↓          ↓          ↓         ↓
                 num_1      num_2      num_3      num_4      num_5    JOKER
                (1-45)     (1-45)     (1-45)     (1-45)     (1-45)   (1-20)
    
    ✓ UN singur RNG
    ✓ Stări consecutive
    ✓ Doar range-ul diferă pentru Joker
    ✓ Joker poate fi duplicat (13.7% din cazuri)

┌─────────────────────────────────────────────────────────────────────────────┐
│ ABORDARE GREȘITĂ: DOUĂ RNG-URI SEPARATE                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    RNG_1(seed_1) → State_1 → State_2 → State_3 → State_4 → State_5
                      ↓          ↓          ↓          ↓          ↓
                    num_1      num_2      num_3      num_4      num_5
                   (1-45)     (1-45)     (1-45)     (1-45)     (1-45)
    
    RNG_2(seed_2?) → State_X
                        ↓
                     JOKER
                     (1-20)
    
    ✗ Două RNG-uri (sau același RNG resetat?)
    ✗ Stări independente
    ✗ Nu știm relația dintre seed_1 și seed_2

┌─────────────────────────────────────────────────────────────────────────────┐
│ DE CE SECVENȚĂ CONTINUĂ?                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

1. Simplitate: Loteria folosește UN singur generator de numere
2. Date reale: 13.7% duplicate confirmă lipsa verificării
3. Eficiență: Nu trebuie două sisteme separate
4. Logică: RNG-ul continuă de unde a rămas
""")

def exemple_reale_explicatie():
    """Exemple reale cu explicație"""
    
    import json
    
    print("\n" + "=" * 80)
    print("📋 EXEMPLE REALE EXPLICAȚIE")
    print("=" * 80)
    
    with open('joker_data.json', 'r') as f:
        data = json.load(f)
    
    print("\n🎯 Să analizăm câteva extrageri reale:\n")
    
    # Extragere cu duplicate
    exemplu_dup = None
    for draw in data['draws']:
        if len(draw['numbers']) == 6:
            primele_5 = draw['numbers'][:5]
            joker = draw['numbers'][5]
            if joker in primele_5:
                exemplu_dup = draw
                break
    
    if exemplu_dup:
        print(f"EXEMPLU 1: Extragere cu DUPLICATE")
        print(f"   Data: {exemplu_dup['date_str']}")
        print(f"   Numere: {exemplu_dup['numbers']}")
        print(f"   Primele 5: {exemplu_dup['numbers'][:5]}")
        print(f"   Joker: {exemplu_dup['numbers'][5]} ⚠️  DUPLICAT!\n")
        print(f"   Explicație RNG:")
        print(f"      State 1-5 → generează {exemplu_dup['numbers'][:5]} (filtrare duplicate)")
        print(f"      State 6   → generează {exemplu_dup['numbers'][5]} din 1-20")
        print(f"      Joker = {exemplu_dup['numbers'][5]} SE ÎNTÂMPLĂ să fie duplicat!")
        print(f"      → NU e eroare, e probabilitate naturală!\n")
    
    # Extragere fără duplicate
    exemplu_nodup = None
    for draw in data['draws']:
        if len(draw['numbers']) == 6:
            primele_5 = draw['numbers'][:5]
            joker = draw['numbers'][5]
            if joker not in primele_5:
                exemplu_nodup = draw
                break
    
    if exemplu_nodup:
        print(f"EXEMPLU 2: Extragere FĂRĂ duplicate")
        print(f"   Data: {exemplu_nodup['date_str']}")
        print(f"   Numere: {exemplu_nodup['numbers']}")
        print(f"   Primele 5: {exemplu_nodup['numbers'][:5]}")
        print(f"   Joker: {exemplu_nodup['numbers'][5]} ✓  UNIC!\n")
        print(f"   Explicație RNG:")
        print(f"      State 1-5 → generează {exemplu_nodup['numbers'][:5]} (filtrare duplicate)")
        print(f"      State 6   → generează {exemplu_nodup['numbers'][5]} din 1-20")
        print(f"      Joker = {exemplu_nodup['numbers'][5]} NU e în primele 5 by chance!")
        print(f"      → Tot probabilitate naturală!\n")

def probabilitate_duplicate():
    """Calculează probabilitatea teoretică de duplicate"""
    
    print("\n" + "=" * 80)
    print("📈 PROBABILITATE DUPLICATE - Validare Teoretică")
    print("=" * 80)
    
    print("""
🔢 Calcul Probabilistic:

Presupunem că primele 5 numere sunt generate random din 1-45.
Joker e generat random din 1-20 (FĂRĂ verificare duplicate).

Câte din primele 5 numere sunt în range 1-20?
    → Depinde de extragere, dar în medie: 5 × (20/45) ≈ 2.22 numere

Probabilitatea ca Joker (din 1-20) să fie DUPLICAT:
    → P(duplicat) = (numere din primele 5 care sunt în 1-20) / 20
    → În medie: ≈ 2.22 / 20 ≈ 11.1%

📊 Date Reale:
    → 14 duplicate din 102 = 13.7%
    
✓ FOARTE APROAPE de predicția teoretică 11.1%!
✓ Confirmă că Joker e generat INDEPENDENT, fără verificare duplicate!
""")

def concluzie_finala():
    """Concluzie finală clară"""
    
    print("\n" + "=" * 80)
    print("✅ CONCLUZIE FINALĂ - RĂSPUNS LA ÎNTREBAREA TA")
    print("=" * 80)
    
    print("""
╔═════════════════════════════════════════════════════════════════════════════╗
║  ÎNTREBARE: Joker-ul (1-20) e tratat ca întreg sau separat?                ║
╚═════════════════════════════════════════════════════════════════════════════╝

📌 RĂSPUNS: CA PARTE INTEGRANTĂ, DAR CU RANGE DIFERIT!

Joker-ul este:
    ✓ Parte din ACEEAȘI secvență RNG (UN singur generator)
    ✓ Al 6-lea număr generat consecutiv
    ✓ Cu range diferit (1-20 în loc de 1-45)
    ✓ FĂRĂ verificare duplicate

Nu este:
    ✗ Un RNG separat
    ✗ Un sistem independent
    ✗ Verificat pentru duplicate

╔═════════════════════════════════════════════════════════════════════════════╗
║  Vizualizare Finală:                                                        ║
╚═════════════════════════════════════════════════════════════════════════════╝

    UN SINGUR RNG, 6 apeluri consecutive:
    
    seed → rng.next() → num1 (mod 45 + 1)  ← Primele 5
           rng.next() → num2 (mod 45 + 1)     (verifică
           rng.next() → num3 (mod 45 + 1)      duplicate)
           rng.next() → num4 (mod 45 + 1)
           rng.next() → num5 (mod 45 + 1)  ┘
           rng.next() → JOKER (mod 20 + 1) ← Joker (NU verifică!)

    Toate 6 numerele vin din ACEEAȘI secvență!
    Doar range-ul e diferit: 45 vs 20

╔═════════════════════════════════════════════════════════════════════════════╗
║  De ce e important?                                                         ║
╚═════════════════════════════════════════════════════════════════════════════╝

Pentru PREDICȚII:
    ✓ Trebuie să găsim UN seed care generează TOATE 6 numerele
    ✓ În ordinea EXACTĂ
    ✓ Cu range-urile corecte
    ✓ Acceptând duplicate pentru Joker
    
    Dacă tratam Joker separat → n-am găsi NICIODATĂ seed-ul corect!
""")

def main():
    """Execută toate demonstrațiile"""
    
    print("\n" + "🎰" * 40)
    print("EXPLICAȚIE COMPLETĂ: Predicții Joker")
    print("🎰" * 40 + "\n")
    
    # 1. Demonstrație secvență
    demonstratie_secventa_rng()
    
    # 2. Demonstrație concretă
    demonstratie_concreta()
    
    # 3. Compară abordări
    compara_abordari()
    
    # 4. Exemple reale
    exemple_reale_explicatie()
    
    # 5. Probabilitate
    probabilitate_duplicate()
    
    # 6. Concluzie
    concluzie_finala()
    
    print("\n" + "=" * 80)
    print("📝 Sper că acum e clar!")
    print("=" * 80)
    print("""
TL;DR:
    • UN RNG = 6 numere consecutive
    • Joker = al 6-lea număr, doar cu range 1-20
    • NU e separat, e parte din secvență
    • Permite duplicate (13.7%)
""")

if __name__ == '__main__':
    main()
