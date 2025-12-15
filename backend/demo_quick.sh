#!/bin/bash
# Quick Demo - Exemplu rapid de utilizare

echo "============================================================="
echo "     DEMO RAPID - Sistem Analiză Loto 5/40"
echo "============================================================="
echo ""

# 1. Extrage date pentru 2024
echo "[1/4] Extragere date pentru anul 2024..."
python3 /app/backend/loto_scraper.py --year 2024 --output /app/backend/loto_demo.json

if [ $? -ne 0 ]; then
    echo "Eroare la extragere!"
    exit 1
fi

echo ""
echo "============================================================="
echo ""

# 2. Analiză statistică scurtă
echo "[2/4] Analiză statistică..."
python3 /app/backend/loto_analyzer.py --input /app/backend/loto_demo.json --top 5 2>&1 | head -60

echo ""
echo "============================================================="
echo ""

# 3. Demo RNG (primele 40 linii)
echo "[3/4] Demonstrație RNG (reverse engineering educațional)..."
python3 /app/backend/rng_demo.py --demo 2>&1 | head -40

echo ""
echo "... (vezi output complet cu: python3 rng_demo.py --demo)"
echo ""
echo "============================================================="
echo ""

# 4. Generare combinații
echo "[4/4] Generare 3 combinații cu strategii diferite..."
echo ""
echo "Strategie BALANCED:"
python3 /app/backend/predictor.py --strategy balanced --count 1 --data /app/backend/loto_demo.json 2>&1 | grep -A 2 "Combinatii generate"

echo ""
echo "Strategie HOT (numere fierbinți):"
python3 /app/backend/predictor.py --strategy hot --count 1 --data /app/backend/loto_demo.json 2>&1 | grep -A 2 "Combinatii generate"

echo ""
echo "Strategie MIXED:"
python3 /app/backend/predictor.py --strategy mixed --count 1 --data /app/backend/loto_demo.json 2>&1 | grep -A 2 "Combinatii generate"

echo ""
echo "============================================================="
echo "                    DEMO FINALIZAT!"
echo "============================================================="
echo ""
echo "Pentru utilizare completă, vezi:"
echo "  - README_LOTO.md (documentație completă)"
echo "  - bash run_all.sh (wizard interactiv)"
echo ""
echo "Comenzi individuale:"
echo "  python3 loto_scraper.py --help"
echo "  python3 loto_analyzer.py --help"
echo "  python3 rng_demo.py --help"
echo "  python3 predictor.py --help"
echo ""
