#!/bin/bash
# Quick Analyze Script - Analiză rapidă pentru orice loterie
#
# Utilizare:
#   ./quick_analyze.sh 6-49 2024
#   ./quick_analyze.sh joker all
#   ./quick_analyze.sh 5-40 2024,2023,2022

set -e

# Verifică argumentele
if [ $# -lt 2 ]; then
    echo "Utilizare: $0 <lottery_type> <year>"
    echo ""
    echo "Lottery types: 5-40, 6-49, joker"
    echo "Years: 2024, 2024,2023, all"
    echo ""
    echo "Exemple:"
    echo "  $0 6-49 2024"
    echo "  $0 joker all"
    echo "  $0 5-40 2024,2023,2022"
    exit 1
fi

LOTTERY=$1
YEAR=$2
DATA_FILE="${LOTTERY}_data.json"

echo "======================================================================"
echo "  QUICK ANALYZE - ${LOTTERY^^}"
echo "======================================================================"
echo ""

# Pasul 1: Extrage date
echo "📥 PASUL 1: Extragere date istorice..."
echo ""
python3 unified_lottery_scraper.py --lottery "$LOTTERY" --year "$YEAR" --output "$DATA_FILE"

# Verifică dacă extragerea a reușit
if [ ! -f "$DATA_FILE" ]; then
    echo ""
    echo "❌ EROARE: Extragerea datelor a eșuat!"
    exit 1
fi

echo ""
echo "======================================================================"
echo ""

# Pasul 2: Quick test pentru identificare rapidă
echo "⚡ PASUL 2: Quick Test (4 RNG-uri rapide)..."
echo ""
python3 unified_pattern_finder.py \
    --lottery "$LOTTERY" \
    --input "$DATA_FILE" \
    --quick-test \
    --min-matches 3 \
    --success-threshold 0.65

echo ""
echo "======================================================================"
echo ""
echo "✅ ANALIZĂ COMPLETĂ!"
echo ""
echo "Fișiere generate:"
echo "  - Date: $DATA_FILE"
echo "  - Rezultate: ${LOTTERY}_pragmatic_results.json"
echo ""
echo "Pentru analiză completă (toate 18 RNG-uri):"
echo "  python3 unified_pattern_finder.py --lottery $LOTTERY --input $DATA_FILE"
echo ""
