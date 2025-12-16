#!/bin/bash
# Script pentru investigarea xorshift_simple pe perioada 2010-2025

echo "=================================================="
echo "  INVESTIGAȚIE XORSHIFT_SIMPLE (2010-2025)"
echo "=================================================="
echo ""

# Verifică dacă există datele
if [ ! -f "5-40_data.json" ]; then
    echo "⚠️  Datele nu există. Scraping în curs..."
    echo ""
    python3 unified_lottery_scraper.py --lottery 5-40 --year all
    echo ""
fi

echo "🚀 Start investigație..."
echo ""
python3 predict_xorshift.py --lottery 5-40 --start-year 2010 --end-year 2025

