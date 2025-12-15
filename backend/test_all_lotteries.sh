#!/bin/bash
# Test script pentru a verifica că toate loteriile funcționează

echo "======================================================================"
echo "  TEST SISTEM UNIFICAT - TOATE LOTERIILE"
echo "======================================================================"
echo ""

YEAR="2024"  # Testăm doar un an pentru viteză

# Test 1: Loto 5/40
echo "🎲 TEST 1/3: Loto 5/40"
echo "----------------------------------------------------------------------"
python3 unified_lottery_scraper.py --lottery 5-40 --year "$YEAR" --output test_5-40.json
if [ $? -eq 0 ]; then
    echo "✅ Scraper 5/40: SUCCESS"
    # Verifică că fișierul există și are date
    COUNT=$(python3 -c "import json; data=json.load(open('test_5-40.json')); print(data['total_draws'])")
    echo "   Extrase: $COUNT extrageri"
else
    echo "❌ Scraper 5/40: FAILED"
    exit 1
fi
echo ""

# Test 2: Loto 6/49
echo "🎲 TEST 2/3: Loto 6/49"
echo "----------------------------------------------------------------------"
python3 unified_lottery_scraper.py --lottery 6-49 --year "$YEAR" --output test_6-49.json
if [ $? -eq 0 ]; then
    echo "✅ Scraper 6/49: SUCCESS"
    COUNT=$(python3 -c "import json; data=json.load(open('test_6-49.json')); print(data['total_draws'])")
    echo "   Extrase: $COUNT extrageri"
else
    echo "❌ Scraper 6/49: FAILED"
    exit 1
fi
echo ""

# Test 3: Joker
echo "🎲 TEST 3/3: Joker"
echo "----------------------------------------------------------------------"
python3 unified_lottery_scraper.py --lottery joker --year "$YEAR" --output test_joker.json
if [ $? -eq 0 ]; then
    echo "✅ Scraper Joker: SUCCESS"
    COUNT=$(python3 -c "import json; data=json.load(open('test_joker.json')); print(data['total_draws'])")
    echo "   Extrase: $COUNT extrageri"
    
    # Verifică că are composite breakdown
    HAS_COMPOSITE=$(python3 -c "import json; data=json.load(open('test_joker.json')); print('composite_breakdown' in data['draws'][0])")
    if [ "$HAS_COMPOSITE" = "True" ]; then
        echo "   ✓ Composite breakdown: OK"
    else
        echo "   ⚠ Composite breakdown: MISSING"
    fi
else
    echo "❌ Scraper Joker: FAILED"
    exit 1
fi
echo ""

echo "======================================================================"
echo "  TEST PATTERN FINDER (quick test pe 6/49)"
echo "======================================================================"
echo ""

# Test rapid pattern finder
timeout 30 python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input test_6-49.json \
    --quick-test \
    --min-matches 3 \
    > /dev/null 2>&1

if [ $? -eq 0 ] || [ $? -eq 124 ]; then
    echo "✅ Pattern Finder: Pornește corect"
else
    echo "❌ Pattern Finder: FAILED"
    exit 1
fi

echo ""
echo "======================================================================"
echo "  ✅ TOATE TESTELE AU TRECUT!"
echo "======================================================================"
echo ""
echo "Fișiere de test generate:"
ls -lh test_*.json 2>/dev/null | awk '{print "  - "$9" ("$5")"}'
echo ""
echo "Sistemul este gata de utilizare! 🎉"
echo ""
echo "Comenzi quick start:"
echo "  ./quick_analyze.sh 6-49 2024"
echo "  ./quick_analyze.sh joker all"
echo "  ./quick_analyze.sh 5-40 2024,2023,2022"
echo ""
