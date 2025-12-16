#!/bin/bash
# TEST COMPLET - Exact cum ai cerut!
# 1. Fiecare an până la 2010
# 2. Apoi 2 ani
# 3. Apoi 3 ani
# 4. Apoi 5 ani
# 5. Apoi 10 ani
# 6. La FINAL: TOȚI anii deodată!

LOTTERY="5-40"  # Schimbă cu 6-49 sau joker
DATA_FILE="${LOTTERY}_data.json"

echo "============================================"
echo "TEST COMPLET - ${LOTTERY}"
echo "Început: $(date)"
echo "============================================"
echo ""

# PHASE 1: Fiecare an până la 2010
echo "PHASE 1: Testing fiecare an până la 2010..."
for year in {1995..2010}; do
  echo "  Testing year $year..."
  python3 -c "
import json
data = json.load(open('$DATA_FILE'))
draws = [d for d in data['draws'] if d['year'] == $year]
new = dict(data)
new['draws'] = draws
new['total_draws'] = len(draws)
json.dump(new, open('temp.json', 'w'))
  "
  python3 unified_pattern_finder.py --lottery $LOTTERY --input temp.json --quick-test > result_year_$year.txt 2>&1
  echo "    Done: $year"
done
echo "✅ Phase 1 complete!"
echo ""

# PHASE 2: 2 ani combinați
echo "PHASE 2: Testing 2 ani combinați..."
for year in {1995..2024}; do
  y2=$((year+1))
  if [ $y2 -le 2025 ]; then
    echo "  Testing $year + $y2..."
    python3 -c "
import json
data = json.load(open('$DATA_FILE'))
draws = [d for d in data['draws'] if d['year'] in [$year,$y2]]
new = dict(data)
new['draws'] = draws
new['total_draws'] = len(draws)
json.dump(new, open('temp.json', 'w'))
    "
    python3 unified_pattern_finder.py --lottery $LOTTERY --input temp.json --quick-test > result_2years_${year}_${y2}.txt 2>&1
    echo "    Done: $year-$y2"
  fi
done
echo "✅ Phase 2 complete!"
echo ""

# PHASE 3: 3 ani combinați
echo "PHASE 3: Testing 3 ani combinați..."
for year in {1995..2023}; do
  y2=$((year+1))
  y3=$((year+2))
  echo "  Testing $year + $y2 + $y3..."
  python3 -c "
import json
data = json.load(open('$DATA_FILE'))
draws = [d for d in data['draws'] if d['year'] in [$year,$y2,$y3]]
new = dict(data)
new['draws'] = draws
new['total_draws'] = len(draws)
json.dump(new, open('temp.json', 'w'))
  "
  python3 unified_pattern_finder.py --lottery $LOTTERY --input temp.json --quick-test > result_3years_${year}_${y3}.txt 2>&1
  echo "    Done: $year-$y3"
done
echo "✅ Phase 3 complete!"
echo ""

# PHASE 4: 5 ani combinați
echo "PHASE 4: Testing 5 ani combinați..."
for year in {1995..2021}; do
  y2=$((year+1))
  y3=$((year+2))
  y4=$((year+3))
  y5=$((year+4))
  echo "  Testing $year până la $y5 (5 ani)..."
  python3 -c "
import json
data = json.load(open('$DATA_FILE'))
draws = [d for d in data['draws'] if d['year'] in [$year,$y2,$y3,$y4,$y5]]
new = dict(data)
new['draws'] = draws
new['total_draws'] = len(draws)
json.dump(new, open('temp.json', 'w'))
  "
  python3 unified_pattern_finder.py --lottery $LOTTERY --input temp.json --quick-test > result_5years_${year}_${y5}.txt 2>&1
  echo "    Done: $year-$y5"
done
echo "✅ Phase 4 complete!"
echo ""

# PHASE 5: 10 ani combinați
echo "PHASE 5: Testing 10 ani combinați..."
for year in {1995..2016}; do
  y10=$((year+9))
  echo "  Testing $year până la $y10 (10 ani)..."
  years_list=$(seq -s, $year $y10)
  python3 -c "
import json
data = json.load(open('$DATA_FILE'))
years = list(range($year, $y10+1))
draws = [d for d in data['draws'] if d['year'] in years]
new = dict(data)
new['draws'] = draws
new['total_draws'] = len(draws)
json.dump(new, open('temp.json', 'w'))
  "
  python3 unified_pattern_finder.py --lottery $LOTTERY --input temp.json --quick-test > result_10years_${year}_${y10}.txt 2>&1
  echo "    Done: $year-$y10"
done
echo "✅ Phase 5 complete!"
echo ""

# PHASE 6: TOȚI ANII DEODATĂ! 🎯
echo "============================================"
echo "PHASE 6 (FINAL): Testing TOȚI ANII DEODATĂ!"
echo "============================================"
echo ""
python3 unified_pattern_finder.py --lottery $LOTTERY --input $DATA_FILE --quick-test > result_ALL_YEARS_TOGETHER.txt 2>&1
echo "✅ Phase 6 (FINAL) complete!"
echo ""

# Generează raport final
echo "============================================"
echo "GENERARE RAPORT FINAL"
echo "============================================"
echo ""

python3 << 'PYTHON_SCRIPT'
import re
from glob import glob

print("\n" + "="*70)
print("RAPORT FINAL - TOP 20 REZULTATE")
print("="*70 + "\n")

results = []

for file in glob('result_*.txt'):
    with open(file, 'r') as f:
        content = f.read()
    
    # Caută success rates
    matches = re.findall(r'(\w+).*?success rate.*?(\d+\.\d+)%', content, re.IGNORECASE)
    
    for rng, rate in matches:
        rate_float = float(rate) / 100
        if rate_float > 0.3:  # Doar peste 30%
            test_name = file.replace('result_', '').replace('.txt', '')
            results.append({
                'test': test_name,
                'rng': rng,
                'rate': rate_float
            })

# Sort by rate
results.sort(key=lambda x: x['rate'], reverse=True)

print("Top 20 cele mai bune rezultate:\n")
for i, r in enumerate(results[:20], 1):
    print(f"{i:2}. {r['test']:50s} | {r['rng']:15s} | {r['rate']:.1%}")

print("\n" + "="*70)
print(f"Total teste cu rate > 30%: {len(results)}")
print(f"Total teste cu rate > 65%: {sum(1 for r in results if r['rate'] >= 0.65)}")
print("="*70 + "\n")

PYTHON_SCRIPT

echo ""
echo "============================================"
echo "✅ TOATE TESTELE COMPLETE!"
echo "Terminat: $(date)"
echo "============================================"
echo ""
echo "Vezi rezultatele în fișierele result_*.txt"
echo "Cel mai important: result_ALL_YEARS_TOGETHER.txt"
echo ""
