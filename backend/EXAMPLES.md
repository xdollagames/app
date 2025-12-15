# 📖 Exemple de Utilizare - Sistem Unificat Loterii

## 🎯 Exemple Rapide

### Exemplul 1: Test Rapid pe Loto 6/49

```bash
cd /app/backend

# Quick analyze - simplu și rapid
./quick_analyze.sh 6-49 2024
```

**Ce face**:
1. Extrage date pentru 2024
2. Rulează quick test (4 RNG-uri rapide)
3. Generează predicții dacă găsește pattern-uri

**Output**:
- `6-49_data.json` - datele istorice
- `6-49_pragmatic_results.json` - rezultatele analizei

---

### Exemplul 2: Analiză Completă pe Joker

```bash
cd /app/backend

# Pasul 1: Extrage toate datele (1995-2025)
python3 unified_lottery_scraper.py --lottery joker --year all

# Pasul 2: Analiză cu TOATE 18 RNG-urile
python3 unified_pattern_finder.py \
    --lottery joker \
    --input joker_data.json \
    --min-matches 3 \
    --success-threshold 0.70

# Pasul 3: Vezi rezultatele
cat joker_pragmatic_results.json | python3 -m json.tool
```

**Timp estimat**: 
- Scraping: ~5-10 minute (pentru 30 ani de date)
- Analiză: ~2-3 ore (depinde de CPU)

---

### Exemplul 3: Comparație între Loterii

```bash
cd /app/backend

# Extrage date pentru toate cele 3 loterii (ultimii 5 ani)
python3 unified_lottery_scraper.py --lottery 5-40 --year 2024,2023,2022,2021,2020
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024,2023,2022,2021,2020
python3 unified_lottery_scraper.py --lottery joker --year 2024,2023,2022,2021,2020

# Quick test pe toate
python3 unified_pattern_finder.py --lottery 5-40 --input 5-40_data.json --quick-test
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_data.json --quick-test
python3 unified_pattern_finder.py --lottery joker --input joker_data.json --quick-test

# Compară rezultatele
echo "=== COMPARAȚIE SUCCÈS RATES ==="
for lottery in 5-40 6-49 joker; do
    echo -n "$lottery: "
    python3 -c "import json; data=json.load(open('${lottery}_pragmatic_results.json')); results=data.get('results',{}); print(max([r['success_rate'] for r in results.values()]) if results else 0)"
done
```

---

### Exemplul 4: Test Specific RNG

Dacă ai o suspiciune despre un anumit tip de RNG:

```bash
cd /app/backend

# Testează doar Mersenne Twister și LCG-uri
python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input 6-49_data.json \
    --rng-types mersenne lcg_glibc lcg_minstd lcg_weak \
    --min-matches 4 \
    --success-threshold 0.75
```

---

### Exemplul 5: Ultimii 10 Ani cu Analiză Intensivă

```bash
cd /app/backend

# Extrage ultimii 10 ani
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024,2023,2022,2021,2020,2019,2018,2017,2016,2015

# Analiză cu search size mare pentru precizie
python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input 6-49_data.json \
    --search-size 5000000 \
    --min-matches 3 \
    --seed-range 0 50000000 \
    --workers 16
```

**⚠️ Atenție**: Foarte computațional intensiv! Poate dura 6-8 ore.

---

## 🔍 Exemple de Interpretare Rezultate

### Scenario 1: Success Rate < 65%

```
lcg_weak      : 23.4% (120/512)
xorshift32    : 31.2% (159/512)
mersenne      : 18.7% (95/512)
```

**Interpretare**:
- ❌ Niciun RNG nu atinge threshold-ul
- ✅ **CONFIRMARE**: Loteria E aleatoare (fizică)
- ✅ NU e generată de software
- ✅ Impredictibilă

**Concluzie**: Exact ce te aștepți de la o loterie REALĂ! 🎉

---

### Scenario 2: Success Rate >= 70% (teoretic, foarte improbabil)

```
lcg_weak      : 78.3% (401/512)  
  └─ linear: R²=0.923
xorshift32    : 12.1% (62/512)
```

**Interpretare**:
- ⚠️ UN RNG are success rate foarte mare
- ⚠️ Pattern găsit în seeds (R²=0.923)
- 🔴 **SUSPICIUNE**: Posibil generată de RNG software
- 🔴 Potențial predictibilă

**Concluzie**: La o loterie REALĂ acest scenariu NU se va întâmpla! Dacă se întâmplă = probleme grave cu aleatoritatea.

---

### Scenario 3: Joker - Composite Analysis

```json
{
  "lottery_type": "joker",
  "draws": [{
    "numbers": [3, 14, 26, 41, 7, 8],
    "composite_breakdown": {
      "part_1": {
        "numbers": [3, 14, 26, 41, 7],
        "range": "1-45"
      },
      "part_2": {
        "numbers": [8],
        "range": "1-20"
      }
    }
  }]
}
```

**Predicții Generate**:
```
1. Method: median_seed
   RNG: lcg_weak
   Seed: 4,523,891
   🎲 Prediction:
      Partea 1 (5 din 1-45): [7, 15, 23, 31, 42]
      Partea 2 (1 din 1-20): [13]
```

**Cum să testezi**:
1. Așteaptă următoarea extragere reală
2. Compară cu predicția
3. Calculează matches
4. Repetă pentru mai multe extrageri

---

## 📊 Exemple de Output JSON

### Loto 6/49 - Rezultate Analiză

```json
{
  "lottery_type": "6-49",
  "lottery_name": "Loto 6/49",
  "config": {
    "min_matches": 3,
    "success_threshold": 0.65,
    "total_draws": 512
  },
  "results": {
    "lcg_weak": {
      "success_rate": 0.234,
      "success_count": 120,
      "total_draws": 512,
      "avg_matches": 2.8,
      "patterns": []
    }
  },
  "predictions": []
}
```

### Joker - Cu Statistici pe Componente

```json
{
  "lottery_type": "joker",
  "total_draws": 512,
  "draws": [
    {
      "date": "2024-12-15",
      "numbers": [3, 14, 26, 41, 7, 8],
      "composite_breakdown": {
        "part_1": {
          "numbers": [3, 14, 26, 41, 7],
          "range": "1-45",
          "description": "5 din 1-45"
        },
        "part_2": {
          "numbers": [8],
          "range": "1-20",
          "description": "1 din 1-20"
        }
      }
    }
  ]
}
```

---

## 🛠️ Exemple de Troubleshooting

### Problem: Scraper nu găsește date

```bash
# Verifică manual URL-ul în browser
firefox http://noroc-chior.ro/Loto/6-din-49/arhiva-rezultate.php?Y=2024

# Verifică conectivitatea
curl -I http://noroc-chior.ro/Loto/6-din-49/arhiva-rezultate.php

# Încearcă alt an
python3 unified_lottery_scraper.py --lottery 6-49 --year 2023
```

---

### Problem: Analiză prea lentă

```bash
# Opțiunea 1: Folosește quick test
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_data.json --quick-test

# Opțiunea 2: Reduce search size
python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input 6-49_data.json \
    --search-size 500000 \
    --workers 16

# Opțiunea 3: Testează doar câteva RNG-uri
python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input 6-49_data.json \
    --rng-types lcg_weak xorshift32 mersenne
```

---

### Problem: "Unknown lottery type"

```bash
# GREȘIT
python3 unified_lottery_scraper.py --lottery loto649
python3 unified_lottery_scraper.py --lottery 649

# CORECT (folosește exact aceste valori)
python3 unified_lottery_scraper.py --lottery 6-49
python3 unified_lottery_scraper.py --lottery joker
python3 unified_lottery_scraper.py --lottery 5-40

# Vezi toate opțiunile disponibile
python3 lottery_config.py
```

---

## 📈 Exemple de Workflow Real

### Workflow 1: Cercetător Începător

```bash
cd /app/backend

# Zi 1: Colectare date și test rapid
./quick_analyze.sh 6-49 2024
./quick_analyze.sh joker 2024

# Zi 2: Analiză completă pe date mai multe
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024,2023,2022
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_data.json

# Zi 3: Comparație și concluzii
# Compară rezultatele, trage concluzii
```

---

### Workflow 2: Analiză Aprofundată

```bash
cd /app/backend

# Săptămâna 1: Colectare date complete
python3 unified_lottery_scraper.py --lottery 6-49 --year all
python3 unified_lottery_scraper.py --lottery joker --year all

# Săptămâna 2: Analiză exhaustivă
python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input 6-49_data.json \
    --search-size 10000000 \
    --seed-range 0 100000000

python3 unified_pattern_finder.py \
    --lottery joker \
    --input joker_data.json \
    --search-size 10000000 \
    --seed-range 0 100000000

# Săptămâna 3: Analiză rezultate și raport
# Compară success rates între loterii
# Analizează diferențele
# Scrie concluzii
```

---

### Workflow 3: Test de Validare Continuă

```bash
cd /app/backend

# Lunar: Update date + re-analiză
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_data.json --quick-test

# Compară cu luna anterioară
# Verifică dacă pattern-urile se mențin (nu ar trebui!)
# Confirmă că aleatoritatea continuă
```

---

## 💡 Tips & Tricks

### Tip 1: Paralelizare pentru Mai Multe Loterii

```bash
# Rulează scraping în paralel pentru toate loteriile
python3 unified_lottery_scraper.py --lottery 5-40 --year 2024 &
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024 &
python3 unified_lottery_scraper.py --lottery joker --year 2024 &
wait

echo "Toate scraperele au terminat!"
```

---

### Tip 2: Salvare Output pentru Analiză Ulterioară

```bash
# Salvează tot output-ul într-un fișier
python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input 6-49_data.json \
    2>&1 | tee 6-49_analysis_log.txt

# Apoi analizează log-ul
grep "Success rate" 6-49_analysis_log.txt
```

---

### Tip 3: Automatizare cu Cron

```bash
# Adaugă în crontab pentru update lunar
0 0 1 * * cd /app/backend && ./quick_analyze.sh 6-49 2024 >> /var/log/lottery_analysis.log 2>&1
```

---

### Tip 4: Extragere Statistici Rapide din JSON

```bash
# Total draws
cat 6-49_data.json | python3 -c "import sys,json; print(json.load(sys.stdin)['total_draws'])"

# Primul și ultimul draw
cat 6-49_data.json | python3 -c "import sys,json; d=json.load(sys.stdin)['draws']; print(f'First: {d[0][\"date\"]}  Last: {d[-1][\"date\"]}')"

# Cel mai frecvent număr
cat 6-49_data.json | python3 -c "import sys,json; from collections import Counter; draws=json.load(sys.stdin)['draws']; nums=[n for d in draws for n in d['numbers']]; c=Counter(nums); print(c.most_common(1))"
```

---

## 🎯 Cazuri de Utilizare

### Caz 1: Student / Cercetător

**Obiectiv**: Înțelegere probabilități și aleatoritate

**Workflow**:
1. Quick test pe toate cele 3 loterii
2. Comparare success rates
3. Studiu documentație RNG-uri
4. Concluzii despre aleatoritate

---

### Caz 2: Developer

**Obiectiv**: Testare RNG-uri proprii

**Workflow**:
1. Adaugă propriul RNG în `advanced_rng_library.py`
2. Testează pe date reale de loterie
3. Compară cu RNG-uri existente
4. Validare calitate aleatoritate

---

### Caz 3: Curios

**Obiectiv**: "Chiar nu există pattern?"

**Workflow**:
1. Rulează analiză completă pe toate loteriile
2. Verifică că NICIUN RNG nu are success rate ridicat
3. Confirmare: Nu există pattern predictibil
4. Peace of mind 😊

---

## 📞 Comenzi Utile de Verificare

```bash
# Verifică versiunea Python
python3 --version

# Verifică librăriile instalate
pip3 list | grep -E "(requests|beautifulsoup4|numpy|scipy)"

# Verifică CPU cores disponibile
python3 -c "from multiprocessing import cpu_count; print(f'CPU cores: {cpu_count()}')"

# Verifică spațiu pe disk
df -h /app/backend

# Verifică toate loteriile configurate
python3 lottery_config.py

# Test rapid sistem
cd /app/backend && ./test_all_lotteries.sh
```

---

**Pentru mai multe exemple și documentație completă, vezi**:
- `README_UNIFIED_SYSTEM.md` - Manual complet
- `MIGRATION_GUIDE.md` - Ghid de tranziție
- `/app/IMPLEMENTATION_SUMMARY.md` - Overview implementare

**Succes la analiză!** 🎲✨
