#!/bin/bash
# INVESTIGAȚIE FOCUSATĂ: xorshift_simple din 2010 până în 2025
# Dacă ai 100% pe 2010, trebuie să vedem dacă continuă!

LOTTERY="5-40"  # Schimbă cu loteria ta
DATA_FILE="${LOTTERY}_data.json"

echo "============================================"
echo "🔥 INVESTIGAȚIE XORSHIFT_SIMPLE"
echo "Din 2010 până în 2025"
echo "============================================"
echo ""

# Testează fiecare an individual cu DOAR xorshift_simple
for year in {2010..2025}; do
  echo "Testing year $year cu xorshift_simple..."
  
  python3 -c "
import json
data = json.load(open('$DATA_FILE'))
draws = [d for d in data['draws'] if d['year'] == $year]
new = dict(data)
new['draws'] = draws
new['total_draws'] = len(draws)
json.dump(new, open('temp_$year.json', 'w'))
  "
  
  # Test DOAR cu xorshift_simple
  python3 unified_pattern_finder.py \
    --lottery $LOTTERY \
    --input temp_$year.json \
    --rng-types xorshift_simple \
    --min-matches 4 \
    --search-size 5000000 \
    > investigation_$year.txt 2>&1
  
  # Extract success rate
  success=$(grep "success rate" investigation_$year.txt | grep "xorshift_simple" | head -1 | grep -oP '\d+\.\d+%' || echo "0.0%")
  
  echo "  Year $year: $success"
  echo ""
done

echo "============================================"
echo "ANALIZA REZULTATELOR"
echo "============================================"
echo ""

# Extrage toate success rates
python3 << 'PYTHON_REPORT'
import re
import json
from glob import glob

print("\n" + "="*60)
print("RAPORT XORSHIFT_SIMPLE: 2010-2025")
print("="*60 + "\n")

results = {}

for year in range(2010, 2026):
    file = f'investigation_{year}.txt'
    try:
        with open(file, 'r') as f:
            content = f.read()
        
        # Găsește success rate
        match = re.search(r'xorshift_simple.*?(\d+\.\d+)%', content, re.IGNORECASE)
        if match:
            rate = float(match.group(1))
            results[year] = {
                'success_rate': rate,
                'file': file
            }
            
            # Verifică dacă are predicții
            if 'PREDICTIONS' in content and 'none' not in content.lower():
                results[year]['has_predictions'] = True
            else:
                results[year]['has_predictions'] = False
    except:
        results[year] = {'success_rate': 0.0, 'file': file, 'has_predictions': False}

# Print results
print(f"{'Year':<8} | {'Success Rate':<15} | {'Predictions':<12}")
print("-" * 60)

for year in sorted(results.keys()):
    r = results[year]
    pred = "✓ YES" if r.get('has_predictions') else "✗ No"
    rate_str = f"{r['success_rate']:.1f}%"
    
    # Color code
    if r['success_rate'] >= 80:
        marker = "🔥"
    elif r['success_rate'] >= 65:
        marker = "⚠️"
    else:
        marker = "  "
    
    print(f"{marker} {year:<6} | {rate_str:<15} | {pred:<12}")

print("\n" + "="*60)

# Summary
high_rate_years = [y for y, r in results.items() if r['success_rate'] >= 70]
print(f"\nAni cu success rate >= 70%: {high_rate_years}")
print(f"Ani cu predicții generate: {[y for y, r in results.items() if r.get('has_predictions')]}")

if len(high_rate_years) >= 3:
    print("\n🔥 ATENȚIE: Success rate ridicat pe mai mulți ani!")
    print("   Posibil pattern detectat!")
else:
    print("\n⚠️ Success rate ridicat doar pe câțiva ani")
    print("   Posibilă coincidență statistică")

PYTHON_REPORT

echo ""
echo "============================================"
echo "EXTRAGE SEED-URILE"
echo "============================================"
echo ""

# Extrage seed-urile pentru anii cu success rate ridicat
python3 << 'PYTHON_SEEDS'
import json
import re
from glob import glob

print("\nEXTRAGERE SEEDS din fișierele cu success rate > 65%:\n")

seeds_by_year = {}

for year in range(2010, 2026):
    file = f'investigation_{year}.txt'
    results_file = f'{year}_pragmatic_results.json'  # Dacă există
    
    try:
        with open(file, 'r') as f:
            content = f.read()
        
        # Caută success rate
        match = re.search(r'xorshift_simple.*?(\d+\.\d+)%', content, re.IGNORECASE)
        if match and float(match.group(1)) >= 65:
            print(f"Anul {year} - Success rate: {match.group(1)}%")
            
            # Încearcă să găsească seeds în content
            seed_matches = re.findall(r'seed[:\s]+(\d+)', content, re.IGNORECASE)
            if seed_matches:
                seeds_by_year[year] = [int(s) for s in seed_matches[:10]]  # Max 10 seeds
                print(f"  Seeds găsite: {seeds_by_year[year][:5]}..." if len(seeds_by_year[year]) > 5 else f"  Seeds: {seeds_by_year[year]}")
            else:
                print(f"  Nu s-au găsit seeds în output")
            print()
    except:
        pass

if len(seeds_by_year) >= 2:
    print("\n" + "="*60)
    print("ANALIZĂ PATTERN ÎN SEEDS")
    print("="*60 + "\n")
    
    # Verifică dacă există pattern între seeds
    years_sorted = sorted(seeds_by_year.keys())
    print(f"Ani cu seeds: {years_sorted}\n")
    
    for i in range(len(years_sorted) - 1):
        y1 = years_sorted[i]
        y2 = years_sorted[i + 1]
        if seeds_by_year[y1] and seeds_by_year[y2]:
            seed1 = seeds_by_year[y1][0]
            seed2 = seeds_by_year[y2][0]
            diff = seed2 - seed1
            ratio = seed2 / seed1 if seed1 != 0 else 0
            
            print(f"{y1} → {y2}:")
            print(f"  Seed {y1}: {seed1}")
            print(f"  Seed {y2}: {seed2}")
            print(f"  Diferență: {diff}")
            print(f"  Ratio: {ratio:.4f}")
            print()

PYTHON_SEEDS

echo ""
echo "============================================"
echo "✅ INVESTIGAȚIE COMPLETĂ!"
echo "============================================"
echo ""
echo "Vezi fișierele investigation_*.txt pentru detalii"
echo "Cei mai importanți: investigation_2010.txt până investigation_2025.txt"
echo ""
