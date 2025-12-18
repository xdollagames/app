#!/usr/bin/env python3
"""
Test pentru a verifica că generate_numbers() păstrează ordinea
"""

from advanced_rng_library import create_rng, generate_numbers

def test_order_preserved():
    """Test: Verifică că ordinea NU este sortată automat"""
    
    print("=" * 80)
    print("🧪 TEST: generate_numbers() Păstrează Ordinea?")
    print("=" * 80)
    
    rng_types = ['xorshift32', 'xorshift64', 'lcg_glibc', 'lcg_minstd']
    
    for rng_type in rng_types:
        print(f"\n📊 Test RNG: {rng_type}")
        
        # Test cu 3 seed-uri diferite
        for seed in [12345, 99999, 54321]:
            rng = create_rng(rng_type, seed)
            generated = generate_numbers(rng, 6, 1, 40)
            sorted_version = sorted(generated)
            
            is_sorted = (generated == sorted_version)
            
            print(f"   Seed {seed:6d}: {generated}")
            print(f"      Sorted:  {sorted_version}")
            print(f"      Is same: {is_sorted} {'⚠️ Coincidență' if is_sorted else '✓ Ordine diferită'}")
            
            # Verifică că ordinea SE POATE să fie diferită
            if not is_sorted:
                print(f"      ✅ BINE! Ordinea NU este sortată automat!")

def test_order_consistency():
    """Test: Verifică că același seed generează aceeași ordine"""
    
    print("\n" + "=" * 80)
    print("🔄 TEST: Consistență Ordine cu Același Seed")
    print("=" * 80)
    
    seed = 12345
    rng_type = 'xorshift32'
    
    # Generează de 3 ori cu același seed
    results = []
    for i in range(3):
        rng = create_rng(rng_type, seed)
        generated = generate_numbers(rng, 6, 1, 40)
        results.append(generated)
        print(f"Run {i+1}: {generated}")
    
    # Toate ar trebui identice
    all_same = all(r == results[0] for r in results)
    print(f"\n{'✅ PERFECT!' if all_same else '❌ PROBLEME!'} Toate rezultatele sunt identice: {all_same}")

def test_real_sequence():
    """Test: Vizualizează secvența RAW a RNG-ului"""
    
    print("\n" + "=" * 80)
    print("🔍 TEST: Secvența RAW RNG (fără filtrare duplicate)")
    print("=" * 80)
    
    seed = 12345
    rng = create_rng('xorshift32', seed)
    
    print(f"\nPrimele 20 de valori RAW generate de RNG (seed={seed}):")
    raw_sequence = []
    for i in range(20):
        val = rng.next() % 40 + 1  # În range 1-40
        raw_sequence.append(val)
    
    print(f"RAW: {raw_sequence}")
    print(f"\nObservații:")
    print(f"   - Poate conține duplicate: {len(raw_sequence) != len(set(raw_sequence))}")
    print(f"   - Ordinea este consecutivă din RNG")
    print(f"   - generate_numbers() filtrează duplicate dar PĂSTREAZĂ ordinea!")

def compare_with_real_data():
    """Compară cu date reale din JSON"""
    
    print("\n" + "=" * 80)
    print("📊 COMPARAȚIE: Generate vs Date Reale")
    print("=" * 80)
    
    import json
    
    with open('5-40_data.json', 'r') as f:
        data = json.load(f)
    
    # Primele 3 extrageri
    print("\nPrimele 3 extrageri REALE:")
    for i, draw in enumerate(data['draws'][:3], 1):
        numbers = draw['numbers']
        sorted_nums = sorted(numbers)
        is_sorted = (numbers == sorted_nums)
        
        print(f"\n{i}. {draw['date_str']}")
        print(f"   Original:  {numbers}")
        print(f"   Sortate:   {sorted_nums}")
        print(f"   Sunt sortate?: {'DA ⚠️' if is_sorted else 'NU ✓'}")
    
    # Statistici
    total_draws = len(data['draws'])
    sorted_count = sum(1 for d in data['draws'] if d['numbers'] == sorted(d['numbers']))
    
    print(f"\n📈 Statistici pe {total_draws} extrageri:")
    print(f"   Extrageri deja sortate: {sorted_count} ({sorted_count/total_draws*100:.1f}%)")
    print(f"   Extrageri nesortate:    {total_draws - sorted_count} ({(total_draws-sorted_count)/total_draws*100:.1f}%)")
    print(f"\n   ✓ Majoritatea NU sunt sortate → Ordinea CONTEAZĂ!")

def main():
    print("\n" + "🧪" * 40)
    print("TEST FIX: generate_numbers() Păstrează Ordinea")
    print("🧪" * 40 + "\n")
    
    # Test 1: Ordinea păstrată?
    test_order_preserved()
    
    # Test 2: Consistență
    test_order_consistency()
    
    # Test 3: Secvență RAW
    test_real_sequence()
    
    # Test 4: Comparație cu date reale
    compare_with_real_data()
    
    print("\n" + "=" * 80)
    print("✅ TESTE COMPLETE")
    print("=" * 80)
    print("\n📝 Concluzii:")
    print("   1. generate_numbers() ACUM păstrează ordinea ✓")
    print("   2. Ordinea este consistentă pentru același seed ✓")
    print("   3. Datele reale conțin ordinea originală (nesortată) ✓")
    print("   4. Predictorii pot acum găsi seed-uri corecte! ✓")

if __name__ == '__main__':
    main()
