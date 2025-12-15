#!/bin/bash
# Script complet pentru extragerea, analiza și generarea de combinații Loto 5/40

echo "======================================================================"
echo "          SISTEM COMPLET ANALIZĂ LOTO 5/40                            "
echo "======================================================================"
echo ""
echo "⚠️  DISCLAIMER: Acest sistem este doar pentru scopuri educaționale."
echo "    Loteriile sunt complet aleatorii și imposibil de prezis!"
echo ""
echo "======================================================================"
echo ""

# Verifică dacă Python 3 este instalat
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 nu este instalat!"
    echo "   Instalează cu: sudo apt install python3 python3-pip"
    exit 1
fi

echo "✓ Python 3 găsit: $(python3 --version)"
echo ""

# Instalează dependențe dacă nu sunt instalate
echo "[1/5] Verificare dependențe..."
if ! python3 -c "import bs4" 2>/dev/null; then
    echo "  Instalare beautifulsoup4..."
    pip3 install beautifulsoup4 requests -q
fi
echo "  ✓ Dependențe OK"
echo ""

# Întreabă utilizatorul ce ani să extragă
echo "[2/5] Extragere date Loto 5/40"
echo "----------------------------------------------------------------------"
read -p "Ce ani dorești să extragi? (ex: 2025, 2024,2023, sau 'all'): " years

if [ -z "$years" ]; then
    years="2025"
    echo "  Folosim implicit: 2025"
fi

echo ""
echo "  Extragere în curs..."
python3 /app/backend/loto_scraper.py --year "$years" --output /app/backend/loto_data.json

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Eroare la extragerea datelor!"
    echo "   Verifică conexiunea internet și încearcă din nou."
    exit 1
fi

echo ""
echo "  ✓ Date extrase cu succes!"
echo ""

# Analiză statistică
echo "[3/5] Analiză statistică"
echo "----------------------------------------------------------------------"
python3 /app/backend/loto_analyzer.py --input /app/backend/loto_data.json --top 10

echo ""
read -p "Apasă ENTER pentru a continua cu demonstrația RNG..."
echo ""

# Demonstrație RNG
echo "[4/5] Demonstrație educațională - Reverse Engineering RNG"
echo "----------------------------------------------------------------------"
python3 /app/backend/rng_demo.py --demo

echo ""
read -p "Apasă ENTER pentru a genera combinații..."
echo ""

# Generare combinații
echo "[5/5] Generare combinații"
echo "----------------------------------------------------------------------"
echo "Strategii disponibile:"
echo "  1. frequency    - Numere frecvente"
echo "  2. balanced     - Echilibru par/impar"
echo "  3. hot          - Numere fierbinți"
echo "  4. cold         - Numere reci"
echo "  5. mixed        - Combinație strategii"
echo "  6. random       - Aleatoriu"
echo "  7. all          - Toate strategiile"
echo ""
read -p "Alege strategie (1-7, sau 'all'): " strategy_choice

case $strategy_choice in
    1) strategy="frequency" ;;
    2) strategy="balanced" ;;
    3) strategy="hot" ;;
    4) strategy="cold" ;;
    5) strategy="mixed" ;;
    6) strategy="random" ;;
    7) strategy="all" ;;
    all) strategy="all" ;;
    *) strategy="random" ;;
esac

read -p "Câte combinații? (1-20): " count
if [ -z "$count" ]; then
    count="3"
fi

echo ""
python3 /app/backend/predictor.py --strategy "$strategy" --count "$count" --data /app/backend/loto_data.json

echo ""
echo "======================================================================"
echo "                         FINALIZAT!                                   "
echo "======================================================================"
echo ""
echo "Fișiere generate:"
echo "  • /app/backend/loto_data.json - Date extrase"
echo ""
echo "Pentru a rula din nou:"
echo "  bash /app/backend/run_all.sh"
echo ""
echo "Pentru scripturi individuale:"
echo "  python3 /app/backend/loto_scraper.py --help"
echo "  python3 /app/backend/loto_analyzer.py --help"
echo "  python3 /app/backend/rng_demo.py --help"
echo "  python3 /app/backend/predictor.py --help"
echo ""
echo "Noroc! (dar amintește-te: fiecare combinație are aceleași șanse!) 🍀"
echo ""
