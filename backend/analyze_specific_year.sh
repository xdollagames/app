#!/bin/bash
# Analizează un an specific dintr-un dataset mare

if [ $# -lt 2 ]; then
    echo "Utilizare: $0 <lottery> <year>"
    echo ""
    echo "Exemple:"
    echo "  $0 6-49 2025"
    echo "  $0 joker 2024"
    echo "  $0 5-40 2023"
    exit 1
fi

LOTTERY=$1
YEAR=$2
FULL_DATA="${LOTTERY}_data.json"
YEAR_DATA="${LOTTERY}_${YEAR}_only.json"

echo "======================================================================"
echo "  ANALIZĂ PE AN SPECIFIC: ${LOTTERY} - ${YEAR}"
echo "======================================================================"
echo ""

# Verifică dacă există date complete
if [ ! -f "$FULL_DATA" ]; then
    echo "⚠️  Nu există fișier $FULL_DATA"
    echo "   Scrapuiesc toate datele mai întâi..."
    python3 unified_lottery_scraper.py --lottery "$LOTTERY" --year all
fi

echo "📥 Extracting doar datele pentru anul $YEAR..."
python3 << EOF
import json

# Încarcă toate datele
with open('$FULL_DATA', 'r') as f:
    all_data = json.load(f)

# Filtrează doar anul dorit
year_draws = [d for d in all_data['draws'] if d['year'] == $YEAR]

if not year_draws:
    print(f"❌ Nu s-au găsit date pentru anul $YEAR")
    exit(1)

# Creează fișier nou doar cu anul specific
filtered_data = {
    'lottery_type': all_data['lottery_type'],
    'lottery_name': all_data['lottery_name'],
    'config': all_data['config'],
    'total_draws': len(year_draws),
    'years': [$YEAR],
    'extracted_at': all_data['extracted_at'],
    'filtered_for_year': $YEAR,
    'draws': year_draws
}

with open('$YEAR_DATA', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"✅ Extras {len(year_draws)} extrageri pentru anul $YEAR")
print(f"   Salvat în: $YEAR_DATA")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Eroare la extragerea datelor"
    exit 1
fi

echo ""
echo "⚡ Rulare analiză quick test pe anul $YEAR..."
echo ""

python3 unified_pattern_finder.py \
    --lottery "$LOTTERY" \
    --input "$YEAR_DATA" \
    --quick-test \
    --min-matches 3

echo ""
echo "======================================================================"
echo "✅ ANALIZĂ COMPLETĂ!"
echo ""
echo "Fișiere generate:"
echo "  - Date complete: $FULL_DATA"
echo "  - Date $YEAR: $YEAR_DATA"
echo "  - Rezultate: ${LOTTERY}_pragmatic_results.json"
echo ""
echo "Pentru analiză completă (toate RNG-urile):"
echo "  python3 unified_pattern_finder.py --lottery $LOTTERY --input $YEAR_DATA"
echo ""
