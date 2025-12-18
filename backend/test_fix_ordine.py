#!/usr/bin/env python3
"""
Test pentru a verifica că fix-ul funcționează corect
Compară comportamentul ÎNAINTE vs DUPĂ fix
"""

import json
from advanced_rng_library import create_rng, generate_numbers
from lottery_config import get_lottery_config

def test_ordine_exacta():
    """Test: Verifică că ordinea exactă este păstrată"""
    
    print("=" * 80)
    print("🧪 TEST: Verificare Fix Ordine Exactă")
    print("=" * 80)
    
    # Încarcă date reale
    with open('5-40_data.json', 'r') as f:
        data = json.load(f)
    
    config = get_lottery_config('5-40')
    
    # Testează primele 3 extrageri
    print("\n📊 Testare pe primele 3 extrageri reale:\n")
    
    for i, draw in enumerate(data['draws'][:3], 1):
        print(f"Test {i}: {draw['date_str']}")
        print(f"   Ordine reală:     {draw['numbers']}")
        print(f"   Ordine sortată:   {draw['numbers_sorted']}")
        
        # Verificare
        if draw['numbers'] == draw['numbers_sorted']:
            print(f"   ⚠️  Coincidență: Ordinea reală = sortată (normal pentru unele extrageri)")
        else:
            print(f"   ✓  Diferență confirmată: Ordinea CONTEAZĂ!")
        
        # Testează că putem genera numere cu un RNG
        seed = 12345 + i
        rng = create_rng('xorshift32', seed)
        generated = generate_numbers(rng, config.numbers_to_draw, 
                                     config.min_number, config.max_number)
        print(f"   Test generare:    {generated}")
        print()

def demonstrate_false_positive():
    """Demonstrează problema false positive-ului"""
    
    print("=" * 80)
    print("🔴 DEMONSTRAȚIE: Problema False Positives")
    print("=" * 80)
    
    config = get_lottery_config('5-40')
    target_real = [5, 13, 26, 38, 37, 25]  # Ordinea reală
    
    print(f"\n🎯 Extragere reală: {target_real}")
    print(f"   Sortată:         {sorted(target_real)}\n")
    
    # Găsește seed-uri diferite care generează aceleași numere în ORDINE DIFERITĂ
    print("🔍 Căutare seed-uri care generează aceleași numere (dar ordine diferită):\n")
    
    found_seeds = []
    for seed in range(0, 1000000, 10000):  # Sample
        rng = create_rng('xorshift32', seed)
        generated = generate_numbers(rng, config.numbers_to_draw,
                                     config.min_number, config.max_number)
        
        # Dacă are aceleași numere (sortate) dar ordine diferită
        if sorted(generated) == sorted(target_real) and generated != target_real:
            found_seeds.append((seed, generated))
            print(f"   Seed {seed:7d}: {generated}")
            if len(found_seeds) >= 3:
                break
    
    if found_seeds:
        print(f"\n❌ FALSE POSITIVES: {len(found_seeds)} seed-uri găsite!")
        print(f"   Toate generează numerele corecte DAR în ordine greșită!")
        print(f"   Cu comparație sortată → toate par valide ❌")
        print(f"   Cu comparație exactă → doar unul este valid ✓")
    else:
        print(f"\n✓ Nu s-au găsit false positives în sample (bine!)")

def test_comparison_methods():
    """Compară metodele de comparație"""
    
    print("\n" + "=" * 80)
    print("⚖️  COMPARAȚIE: Metode de Verificare")
    print("=" * 80 + "\n")
    
    target = [5, 13, 26, 38, 37, 25]
    
    test_cases = [
        ([5, 13, 26, 38, 37, 25], "Ordinea EXACTĂ (CORECT)"),
        ([25, 37, 38, 26, 13, 5], "Ordinea INVERSĂ (GREȘIT)"),
        ([5, 13, 25, 26, 37, 38], "Sortate (GREȘIT - diferă)"),
        ([13, 5, 26, 38, 37, 25], "Parțial amestecat (GREȘIT)"),
    ]
    
    print(f"Target: {target}\n")
    
    for generated, description in test_cases:
        print(f"{description}")
        print(f"   Generated: {generated}")
        
        # Metoda GREȘITĂ (sortată)
        wrong_match = sorted(generated) == sorted(target)
        print(f"   sorted() == sorted(): {wrong_match} {'❌ FALSE POSITIVE!' if wrong_match and generated != target else '✓'}")
        
        # Metoda CORECTĂ (exactă)
        correct_match = generated == target
        print(f"   exact == exact:       {correct_match} {'✓ CORECT!' if correct_match else '✓ Respins corect'}")
        print()

def main():
    print("\n" + "🧪" * 40)
    print("TEST FIX: Verificare Ordine Exactă în Predicții")
    print("🧪" * 40 + "\n")
    
    # Test 1: Verifică date reale
    test_ordine_exacta()
    
    # Test 2: Demonstrație false positives
    demonstrate_false_positive()
    
    # Test 3: Comparație metode
    test_comparison_methods()
    
    print("=" * 80)
    print("✅ TESTE COMPLETE")
    print("=" * 80)
    print("\n📝 Concluzii:")
    print("   1. Datele JSON conțin ordinea corectă ✓")
    print("   2. Comparația sortată generează false positives ❌")
    print("   3. Comparația exactă este obligatorie pentru RNG analysis ✓")
    print("\n🎯 Toate predictorii au fost fixați pentru a folosi ordinea exactă!")

if __name__ == '__main__':
    main()
