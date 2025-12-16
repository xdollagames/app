# 🎯 Exemplu Practic: Analiză Pe An Specific

## Scenariul Tău EXACT

**Vrei**:
1. Să ai TOATE datele (1995-2025) salvate
2. Să analizezi doar 2025 (sau orice alt an)

---

## 🚀 Soluția în 3 Pași

### Pasul 1: Scrapuiește TOATE Datele (o singură dată)

```bash
cd /app/backend

# Scrapuiește tot (1995-2025) - durează ~3-5 minute
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Verifică ce ai
python3 -c "
import json
data = json.load(open('6-49_data.json'))
print(f'✅ {data[\"total_draws\"]} extrageri')
print(f'📅 Ani: {min(data[\"years\"])} - {max(data[\"years\"])}')
"
```

**Output așteptat**:
```
✅ 3247 extrageri
📅 Ani: 1995 - 2025
```

---

### Pasul 2: Analizează DOAR 2025

**Metoda A - Cel Mai Simplu (Scriptul Magic)** ⭐
```bash
# O singură comandă!
./analyze_specific_year.sh 6-49 2025
```

**Ce se întâmplă**:
1. ✅ Citește `6-49_data.json` (toate datele)
2. ✅ Extrage doar extragerile din 2025
3. ✅ Creează `6-49_2025_only.json`
4. ✅ Rulează analiza pe acest fișier
5. ✅ Afișează rezultatele

**Metoda B - Manual**
```bash
# 1. Extrage datele pentru 2025
python3 << 'EOF'
import json

# Încarcă toate datele
with open('6-49_data.json', 'r') as f:
    all_data = json.load(f)

# Filtrează doar 2025
draws_2025 = [d for d in all_data['draws'] if d['year'] == 2025]

# Salvează separat
filtered = {
    'lottery_type': all_data['lottery_type'],
    'lottery_name': all_data['lottery_name'],
    'config': all_data['config'],
    'total_draws': len(draws_2025),
    'years': [2025],
    'extracted_at': all_data['extracted_at'],
    'draws': draws_2025
}

with open('6-49_2025_only.json', 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"✅ Salvat: {len(draws_2025)} extrageri pentru 2025")
EOF

# 2. Analizează
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_2025_only.json --quick-test
```

---

### Pasul 3: Vezi Rezultatele

```bash
# Vezi summary
cat 6-49_pragmatic_results.json | python3 -m json.tool | head -50

# Sau citește direct
python3 << 'EOF'
import json

results = json.load(open('6-49_pragmatic_results.json'))

print("="*60)
print("REZULTATE ANALIZĂ 2025")
print("="*60)

if results.get('results'):
    for rng, data in results['results'].items():
        print(f"\n{rng}:")
        print(f"  Success rate: {data['success_rate']:.1%}")
        print(f"  Matches: {data['success_count']}/{data['total_draws']}")
else:
    print("❌ Niciun RNG nu atinge success threshold")
    print("✅ Confirmare: Loteria e aleatoare!")
EOF
```

---

## 📊 Exemple Complete

### Exemplul 1: Analiză Doar 2025

```bash
cd /app/backend

# Verifică dacă ai datele complete
if [ ! -f "6-49_data.json" ]; then
    echo "Scrapuiesc toate datele..."
    python3 unified_lottery_scraper.py --lottery 6-49 --year all
fi

# Analizează doar 2025
./analyze_specific_year.sh 6-49 2025

# Gata!
```

---

### Exemplul 2: Analiză Multi-An

```bash
cd /app/backend

# Scrapuiește tot (o dată)
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Analizează fiecare an separat
for year in 2025 2024 2023; do
    echo "=== Analizare $year ==="
    ./analyze_specific_year.sh 6-49 $year
    echo ""
done

# Rezultat:
# - 6-49_2025_only.json + analiză
# - 6-49_2024_only.json + analiză
# - 6-49_2023_only.json + analiză
```

---

### Exemplul 3: Comparație 2025 vs 2024

```bash
cd /app/backend

# Setup
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Analizează ambii ani
./analyze_specific_year.sh 6-49 2025
mv 6-49_pragmatic_results.json 6-49_results_2025.json

./analyze_specific_year.sh 6-49 2024
mv 6-49_pragmatic_results.json 6-49_results_2024.json

# Compară
python3 << 'EOF'
import json

def best_rate(file):
    data = json.load(open(file))
    results = data.get('results', {})
    if results:
        return max([r['success_rate'] for r in results.values()])
    return 0

rate_2025 = best_rate('6-49_results_2025.json')
rate_2024 = best_rate('6-49_results_2024.json')

print("\n=== COMPARAȚIE 2025 vs 2024 ===")
print(f"2025: {rate_2025:.1%}")
print(f"2024: {rate_2024:.1%}")

if rate_2025 < 0.3 and rate_2024 < 0.3:
    print("\n✅ Ambii ani: Aleatoriu perfect (normal!)")
elif abs(rate_2025 - rate_2024) < 0.05:
    print("\n✅ Rate similare între ani (consistent aleatoriu)")
else:
    print("\n⚠️ Diferență mare între ani (neobișnuit)")
EOF
```

---

## 🎓 Cazuri de Utilizare Reale

### Caz 1: "Vreau să testez doar ultimul an"

```bash
cd /app/backend

# Quick - doar 2025
python3 unified_lottery_scraper.py --lottery 6-49 --year 2025 --output 6-49_2025.json
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_2025.json --quick-test
```

**Timp**: ~1 minut total

---

### Caz 2: "Am toate datele, vreau să testez fiecare an"

```bash
cd /app/backend

# Deja ai: 6-49_data.json (toate)

# Testează fiecare an
for year in {2020..2025}; do
    ./analyze_specific_year.sh 6-49 $year
    sleep 1
done
```

**Timp**: ~12-15 minute (6 ani × 2 min/an)

---

### Caz 3: "Vreau date complete, dar analizez doar când vreau"

```bash
# Ziua 1: Setup (o singură dată)
cd /app/backend
python3 unified_lottery_scraper.py --lottery 6-49 --year all
tar -czf backup.tar.gz 6-49_data.json

# Ziua 2: Analizează 2025
./analyze_specific_year.sh 6-49 2025

# Săptămâna următoare: Analizează 2024
./analyze_specific_year.sh 6-49 2024

# Luna viitoare: Analizează 2023
./analyze_specific_year.sh 6-49 2023

# etc. - datele rămân salvate MEREU!
```

---

### Caz 4: "Update lunar cu ultimele extrageri"

```bash
# La fiecare lună
cd /app/backend

# Re-scrapuiește doar 2025 (update cu extrageri noi)
python3 unified_lottery_scraper.py --lottery 6-49 --year 2025 --output 6-49_2025_updated.json

# Analizează
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_2025_updated.json --quick-test

# SAU
# Re-scrapuiește TOT (include și 2025 actualizat)
python3 unified_lottery_scraper.py --lottery 6-49 --year all
./analyze_specific_year.sh 6-49 2025
```

---

## 📁 Structura Fișiere După Analiză

```
/app/backend/
│
# Date complete (BAZA ta de date)
├── 6-49_data.json         ← TOATE datele (1995-2025) ~1.2 MB
│
# Date filtrate pe ani
├── 6-49_2025_only.json    ← Doar 2025 ~35 KB
├── 6-49_2024_only.json    ← Doar 2024 ~35 KB
├── 6-49_2023_only.json    ← Doar 2023 ~35 KB
│
# Rezultate analiză
├── 6-49_pragmatic_results.json  ← Ultima analiză
│
# Backup
└── backup.tar.gz          ← Backup date complete
```

---

## 💡 Best Practices

### 1. ✅ Scrapuiește Tot Odată (Setup Inițial)
```bash
# Bun - o dată, pentru totdeauna
python3 unified_lottery_scraper.py --lottery 6-49 --year all
```

### 2. ✅ Analizează Selectiv
```bash
# Analizează doar ce te interesează
./analyze_specific_year.sh 6-49 2025  # Doar 2025
./analyze_specific_year.sh 6-49 2024  # Doar 2024
```

### 3. ✅ Păstrează Date Complete
```bash
# NU șterge niciodată
6-49_data.json  # Păstrează!

# Șterge doar fișierele temporare
rm 6-49_2025_only.json  # Poți re-genera oricând
```

### 4. ✅ Update Periodic
```bash
# Trimestrial sau semestrial
python3 unified_lottery_scraper.py --lottery 6-49 --year all
```

---

## ⚡ Comenzi Rapide (Cheat Sheet)

```bash
# Setup complet
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Analizează an specific
./analyze_specific_year.sh 6-49 2025

# Analizează mai mulți ani
for year in 2025 2024 2023; do ./analyze_specific_year.sh 6-49 $year; done

# Verifică ce ani ai
python3 -c "import json; d=json.load(open('6-49_data.json')); print(d['years'])"

# Număr extrageri pe an
python3 -c "import json; from collections import Counter; d=json.load(open('6-49_data.json')); print(Counter(x['year'] for x in d['draws']))"
```

---

## ✅ Concluzia Ta

**DA! Poți avea exact ce vrei**:

✅ **Baza de date completă** → `6-49_data.json` (toate anii)
✅ **Analiză selectivă** → `./analyze_specific_year.sh 6-49 2025`
✅ **Flexibilitate totală** → Analizezi orice an, oricând
✅ **Date salvate permanent** → Nu se pierd niciodată
✅ **Update ușor** → Re-scrapuiești când vrei date noi

**Comandă magică**:
```bash
./analyze_specific_year.sh 6-49 2025
```

**Gata! 🎉**
