#!/usr/bin/env python3
"""Test cache cu chunks diferite"""

from cpu_only_predictor import cpu_worker_chunked, cache_seed, get_lottery_config

# Salvăm un seed în cache
cache_seed('5-40', '2025-12-14', 'lcg_borland', 100_000_000)
print("✅ Seed 100,000,000 salvat în cache pentru lcg_borland\n")

# Simulăm worker cu chunk DIFERIT de unde e seed-ul
config = get_lottery_config('5-40')
all_targets = [(0, [1, 2, 3, 4, 5, 6], '2025-12-14')]

# Chunk 1: [0 ... 50M) - seed-ul NU e aici
args1 = (all_targets, 'lcg_borland', config, 0, 50_000_000, 999999, '5-40', (0, 4_294_967_296))
print("Test Chunk 1: [0 ... 50M) - seed 100M NU e în acest chunk")
result1 = cpu_worker_chunked(args1)
print(f"  Rezultat: {result1}")
print(f"  A găsit seed din cache? {result1.get(0) == 100_000_000}\n")

# Chunk 2: [90M ... 140M) - seed-ul E aici
args2 = (all_targets, 'lcg_borland', config, 90_000_000, 140_000_000, 999999, '5-40', (0, 4_294_967_296))
print("Test Chunk 2: [90M ... 140M) - seed 100M E în acest chunk")
result2 = cpu_worker_chunked(args2)
print(f"  Rezultat: {result2}")
print(f"  A găsit seed din cache? {result2.get(0) == 100_000_000}\n")

print("="*70)
print("✅ AMBELE chunk-uri ar trebui să găsească seed-ul din cache!")
print("   Seed-ul e valid indiferent de ce chunk procesăm!")
