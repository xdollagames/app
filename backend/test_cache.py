#!/usr/bin/env python3
"""Test cache functionality"""

from cpu_only_predictor import get_cached_seed, cache_seed, load_seeds_cache

# Test 1: Salvare și citire seed găsit
print("Test 1: Salvare seed găsit")
cache_seed('5-40', '2025-12-14', 'test_rng', 123456)
result = get_cached_seed('5-40', '2025-12-14', 'test_rng')
print(f"  Salvat: 123456")
print(f"  Citit: {result}")
print(f"  ✅ Match: {result == 123456}\n")

# Test 2: Salvare și citire NOT_FOUND
print("Test 2: Salvare NOT_FOUND")
cache_seed('5-40', '2025-12-15', 'test_rng', 'NOT_FOUND')
result = get_cached_seed('5-40', '2025-12-15', 'test_rng')
print(f"  Salvat: 'NOT_FOUND'")
print(f"  Citit: {result}")
print(f"  ✅ Match: {result == 'NOT_FOUND'}\n")

# Test 3: Citire inexistent
print("Test 3: Citire seed inexistent")
result = get_cached_seed('5-40', '2025-12-99', 'test_rng')
print(f"  Citit: {result}")
print(f"  ✅ None: {result is None}\n")

# Test 4: Verificare structură cache
print("Test 4: Verificare structură cache")
cache = load_seeds_cache()
print(f"  Versiune: {cache.get('_version')}")
print(f"  Keys: {list(cache.keys())}")
if '5-40' in cache:
    print(f"  5-40 dates: {list(cache['5-40'].keys())}")
    if '2025-12-14' in cache['5-40']:
        print(f"  2025-12-14 RNGs: {cache['5-40']['2025-12-14']}")

print("\n✅ Toate testele completate!")
