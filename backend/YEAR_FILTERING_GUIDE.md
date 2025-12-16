# 📅 Ghid Analiză Pe Ani Specifici

## 🎯 Scenariul Tău

**Vrei**: 
- Să ai TOATE datele (1995-2025) într-un fișier mare
- Să analizezi doar UN an specific (ex: 2025)

**Soluție**: 100% POSIBIL! Mai multe metode:

---

## Metoda 1: Scraping Separat (Cel Mai Simplu) ⭐

### Pasul 1: Scrapuiește Tot
```bash
cd /app/backend

# Scrapuiește TOATE datele (o singură dată)
python3 unified_lottery_scraper.py --lottery 6-49 --year all
# Rezultat: 6-49_data.json (toate datele 1995-2025)
```

### Pasul 2: Scrapuiește Anul Specific
```bash
# Scrapuiește doar 2025 într-un fișier separat
python3 unified_lottery_scraper.py --lottery 6-49 --year 2025 --output 6-49_2025.json

# SAU doar 2024
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024 --output 6-49_2024.json
```

### Pasul 3: Analizează
```bash
# Analizează doar 2025
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_2025.json

# Analizează toate datele
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_data.json

# Analizează doar 2024
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_2024.json
```

**Avantaj**: Simplu, rapid, fișiere separate clare

---

## Metoda 2: Script Automatizat (Recomandat) 🔥

Am creat un script care face totul automat!

### Utilizare
```bash
cd /app/backend

# Analizează doar 2025
./analyze_specific_year.sh 6-49 2025

# Analizează doar 2024
./analyze_specific_year.sh joker 2024

# Analizează doar 2023
./analyze_specific_year.sh 5-40 2023
```

**Ce face scriptul**:
1. Verifică dacă există fișierul complet (6-49_data.json)
2. Dacă nu, îl scrapuiește automat
3. Extrage doar datele pentru anul dorit
4. Rulează analiza pe anul respectiv
5. Salvează rezultatele

**Output**:
```
6-49_data.json         # Toate datele (1995-2025)
6-49_2025_only.json    # Doar 2025
6-49_2024_only.json    # Doar 2024
# etc.
```

---

## Metoda 3: Filtrare Manuală cu Python

### Extract Orice An din Dataset Complet

```bash
cd /app/backend

# Scrapuiește tot (dacă nu ai deja)
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Filtrează manual
python3 << 'EOF'
import json

# Încarcă toate datele
with open('6-49_data.json', 'r') as f:
    all_data = json.load(f)

# Filtrează doar 2025
draws_2025 = [d for d in all_data['draws'] if d['year'] == 2025]

# Creează fișier nou
filtered = {
    'lottery_type': all_data['lottery_type'],
    'lottery_name': all_data['lottery_name'],
    'config': all_data['config'],
    'total_draws': len(draws_2025),
    'years': [2025],
    'extracted_at': all_data['extracted_at'],
    'note': 'Filtered for year 2025 only',
    'draws': draws_2025
}

# Salvează
with open('6-49_2025_only.json', 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"✅ Extracted {len(draws_2025)} draws for 2025")
EOF
```

### Extract Range de Ani

```python
# Filtrează ultimii 3 ani (2023-2025)
draws_recent = [d for d in all_data['draws'] if d['year'] >= 2023]

# Filtrează decada 2010-2019
draws_decade = [d for d in all_data['draws'] if 2010 <= d['year'] <= 2019]
```

---

## Metoda 4: Analiză Directă (Fără Filtrare)

Poți modifica `unified_pattern_finder.py` să accepte un parametru `--filter-year`:

```bash
# VIITOR - nu e implementat încă
python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input 6-49_data.json \
    --filter-year 2025
```

**Notă**: Deocamdată trebuie să filtrezi manual în JSON separat.

---

## 📊 Exemple Concrete

### Exemplul 1: Analiză Multi-An

```bash
cd /app/backend

# Setup (o singură dată)
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Analizează fiecare an separat
for year in 2025 2024 2023 2022 2021; do
    echo "Analizare $year..."
    ./analyze_specific_year.sh 6-49 $year
done

# Rezultat: 5 fișiere cu rezultate separate
# 6-49_2025_only.json
# 6-49_2024_only.json
# etc.
```

### Exemplul 2: Comparație Între Ani

```bash
# Scrapuiește tot
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Analizează 2025
./analyze_specific_year.sh 6-49 2025

# Analizează 2024
./analyze_specific_year.sh 6-49 2024

# Compară success rates
echo "=== COMPARAȚIE 2025 vs 2024 ==="
python3 << 'EOF'
import json

def get_best_rate(file):
    try:
        data = json.load(open(file))
        results = data.get('results', {})
        if results:
            return max([r['success_rate'] for r in results.values()])
        return 0
    except:
        return 0

rate_2025 = get_best_rate('6-49_pragmatic_results.json')  # Latest
rate_2024 = get_best_rate('6-49_pragmatic_results_2024.json')  # Previous

print(f"2025: {rate_2025:.1%}")
print(f"2024: {rate_2024:.1%}")
EOF
```

### Exemplul 3: Setup Complet

```bash
cd /app/backend

echo "=== SETUP COMPLET MULTI-AN ==="

# 1. Scrapuiește TOATE datele pentru toate loteriile
echo "Scraping ALL data..."
python3 unified_lottery_scraper.py --lottery 6-49 --year all &
python3 unified_lottery_scraper.py --lottery joker --year all &
python3 unified_lottery_scraper.py --lottery 5-40 --year all &
wait

echo "✅ Toate datele scrapuite!"

# 2. Analizează anul curent (2025) pentru fiecare
echo "Analizare 2025 pentru toate loteriile..."
./analyze_specific_year.sh 6-49 2025 &
./analyze_specific_year.sh joker 2025 &
./analyze_specific_year.sh 5-40 2025 &
wait

echo "✅ GATA! Ai:"
echo "  - Date complete: *_data.json"
echo "  - Date 2025: *_2025_only.json"
echo "  - Rezultate analiză 2025: *_pragmatic_results.json"
```

---

## 📁 Structura Fișiere Recomandată

```
/app/backend/
├── 6-49_data.json              # TOATE datele (1995-2025)
├── 6-49_2025_only.json         # Doar 2025
├── 6-49_2024_only.json         # Doar 2024
├── 6-49_2023_only.json         # Doar 2023
│
├── joker_data.json             # TOATE datele Joker
├── joker_2025_only.json        # Doar 2025
│
├── 5-40_data.json              # TOATE datele 5/40
├── 5-40_2025_only.json         # Doar 2025
│
└── *_pragmatic_results.json    # Rezultate analiză
```

---

## 💡 Tips & Best Practices

### 1. Păstrează Date Complete
```bash
# ÎNTOTDEAUNA păstrează fișierul complet
6-49_data.json  # NU șterge!

# Creează copii pentru ani specifici
6-49_2025_only.json
6-49_2024_only.json
```

### 2. Naming Convention
```bash
# Bun
6-49_data.json          # Toate
6-49_2025_only.json     # An specific
6-49_recent_3years.json # Range

# Evită
6-49.json               # Nu se știe ce conține
data.json               # Prea generic
```

### 3. Backup
```bash
# Backup date complete
tar -czf lottery_full_backup_$(date +%Y%m%d).tar.gz *_data.json

# Backup analize anuale
tar -czf lottery_2025_analysis.tar.gz *_2025_*.json *_pragmatic_results.json
```

### 4. Update Lunar
```bash
# La fiecare lună, actualizează doar anul curent
python3 unified_lottery_scraper.py --lottery 6-49 --year 2025 --output 6-49_2025_latest.json

# Apoi analizează
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_2025_latest.json
```

---

## 🔍 Verificări Utile

### Verifică Ce Ani Ai în Fișier
```bash
python3 -c "
import json
data = json.load(open('6-49_data.json'))
years = sorted(set(d['year'] for d in data['draws']))
print(f'Ani disponibili: {years[0]}-{years[-1]}')
print(f'Total: {len(years)} ani')
print(f'Total extrageri: {data[\"total_draws\"]}')
"
```

### Număr Extrageri Pe An
```bash
python3 << 'EOF'
import json
from collections import Counter

data = json.load(open('6-49_data.json'))
year_counts = Counter(d['year'] for d in data['draws'])

print("Extrageri pe an:")
for year in sorted(year_counts.keys()):
    print(f"  {year}: {year_counts[year]} extrageri")
EOF
```

### Verifică Range de Date
```bash
python3 -c "
import json
data = json.load(open('6-49_data.json'))
dates = [d['date'] for d in data['draws']]
print(f'Primul: {min(dates)}')
print(f'Ultimul: {max(dates)}')
"
```

---

## 🎯 Workflow Recomandat

### Setup Inițial (o singură dată)
```bash
cd /app/backend

# Scrapuiește TOATE datele
python3 unified_lottery_scraper.py --lottery 6-49 --year all
python3 unified_lottery_scraper.py --lottery joker --year all
python3 unified_lottery_scraper.py --lottery 5-40 --year all

# Backup
tar -czf lottery_complete_archive.tar.gz *_data.json
```

### Analiză Regulată (lunar/săptămânal)
```bash
# Analizează anul curent
./analyze_specific_year.sh 6-49 2025
./analyze_specific_year.sh joker 2025
./analyze_specific_year.sh 5-40 2025

# Compară cu anii precedenți
./analyze_specific_year.sh 6-49 2024
# etc.
```

### Update Date (trimestrial)
```bash
# Re-scrapuiește tot pentru date fresh
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Analizează din nou anul curent
./analyze_specific_year.sh 6-49 2025
```

---

## ❓ FAQ

**Q: Pot avea toate datele și analiza doar 2025?**
✅ **DA! Exact ce face scriptul `analyze_specific_year.sh`**

**Q: Datele pentru 2025 se actualizează automat?**
❌ Nu. Trebuie să re-scrapuiești manual:
```bash
python3 unified_lottery_scraper.py --lottery 6-49 --year 2025 --output 6-49_2025.json
```

**Q: Pot analiza mai mulți ani deodată?**
✅ Da, filtrează mai mulți ani:
```python
draws = [d for d in all_data['draws'] if d['year'] in [2023, 2024, 2025]]
```

**Q: Care metodă e cea mai bună?**
⭐ **Metoda 2 (scriptul automatizat)** - simplu, rapid, automatizat

---

## ✅ Concluzie

**AI CONTROL COMPLET**:
- ✅ Poți avea TOATE datele într-un fișier
- ✅ Poți analiza DOAR un an specific
- ✅ Poți crea fișiere separate pentru fiecare an
- ✅ Datele rămân salvate PERMANENT
- ✅ Flexibilitate maximă!

**Comandă Simplă**:
```bash
./analyze_specific_year.sh 6-49 2025
```

**Gata! 🎉**
