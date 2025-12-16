# 📥 Ghid Complet Scraping Date

## ✅ Scraping Multi-An (DEJA IMPLEMENTAT!)

### Opțiuni Disponibile

#### 1️⃣ Un Singur An
```bash
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024
```

#### 2️⃣ Mai Mulți Ani Specifici
```bash
# Ultimii 3 ani
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024,2023,2022

# Ultimii 5 ani
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024,2023,2022,2021,2020

# Ultimii 10 ani
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024,2023,2022,2021,2020,2019,2018,2017,2016,2015
```

#### 3️⃣ TOȚI Anii (1995-2025) 🔥
```bash
# Extrage TOATĂ arhiva (recomandare: 30 ani de date!)
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Pentru Joker
python3 unified_lottery_scraper.py --lottery joker --year all

# Pentru Loto 5/40
python3 unified_lottery_scraper.py --lottery 5-40 --year all
```

### ⏱️ Timpi Estimați

| Configurație | Extrageri | Timp Estimat |
|-------------|-----------|--------------|
| 1 an | ~100 | 5-10 secunde |
| 3 ani | ~300 | 15-20 secunde |
| 5 ani | ~500 | 25-30 secunde |
| 10 ani | ~1000 | 45-60 secunde |
| ALL (30 ani) | ~3000 | **3-5 minute** |

### 📁 Unde Se Salvează Datele?

**Default**:
```bash
/app/backend/6-49_data.json    # Pentru 6/49
/app/backend/joker_data.json   # Pentru Joker
/app/backend/5-40_data.json    # Pentru 5/40
```

**Custom**:
```bash
python3 unified_lottery_scraper.py \
    --lottery 6-49 \
    --year all \
    --output /path/custom/mele_date_649.json
```

### ✅ Datele Rămân Salvate PERMANENT

**DA! Datele se salvează în fișiere JSON și rămân pe disk:**

```bash
# Scrapuiești o dată
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Fișierul 6-49_data.json e creat și salvat

# Poți folosi datele de câte ori vrei:
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_data.json
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_data.json --quick-test
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_data.json --rng-types mersenne

# Datele NU se șterg, NU se pierd
# Poți copia fișierul, face backup, etc.
```

### 📊 Structura Fișier JSON

```json
{
  "lottery_type": "6-49",
  "lottery_name": "Loto 6/49",
  "config": {
    "numbers_to_draw": 6,
    "min_number": 1,
    "max_number": 49
  },
  "total_draws": 3247,
  "years": [1995, 1996, ..., 2024],
  "extracted_at": "2024-12-15T23:45:00",
  "draws": [
    {
      "date": "1995-01-05",
      "date_str": "Jo, 5 ianuarie 1995",
      "numbers": [7, 15, 23, 31, 38, 45],
      "numbers_sorted": [7, 15, 23, 31, 38, 45],
      "year": 1995,
      "lottery_type": "6-49"
    },
    ... (3247 extrageri)
  ]
}
```

## 🔄 Update Date (Lunar/Săptămânal)

### Opțiune 1: Re-scrape Complet
```bash
# Re-extrage toate datele (include ultimele extrageri)
python3 unified_lottery_scraper.py --lottery 6-49 --year all
```

### Opțiune 2: Scrape Doar Anul Curent
```bash
# Extrage doar 2024
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024 --output 6-49_2024_update.json

# Apoi combină manual cu datele vechi (dacă vrei)
```

### Opțiune 3: Automatizare cu Cron
```bash
# Adaugă în crontab - update lunar
0 0 1 * * cd /app/backend && python3 unified_lottery_scraper.py --lottery 6-49 --year 2024 --output 6-49_latest.json

# Sau săptămânal
0 0 * * 0 cd /app/backend && python3 unified_lottery_scraper.py --lottery 6-49 --year 2024 --output 6-49_latest.json
```

## 💾 Backup și Management Date

### Backup
```bash
# Fă backup la date importante
cp 6-49_data.json 6-49_data_backup_2024-12-15.json

# Sau comprimă
tar -czf lottery_data_backup.tar.gz *_data.json
```

### Verificare Integritate
```bash
# Verifică că JSON-ul e valid
python3 -c "import json; json.load(open('6-49_data.json')); print('✅ Valid JSON')"

# Verifică număr extrageri
python3 -c "import json; data=json.load(open('6-49_data.json')); print(f'Total draws: {data[\"total_draws\"]}')"
```

### Curățare Date Vechi (Dacă Vrei)
```bash
# Șterge datele temporare de test
rm test_*.json

# Păstrează doar datele finale
ls -lh *_data.json
```

## 🎯 Workflow Recomandat

### Prima Dată (Setup Complet)
```bash
# 1. Scrapuiește TOATE datele pentru toate loteriile (o singură dată)
python3 unified_lottery_scraper.py --lottery 5-40 --year all
python3 unified_lottery_scraper.py --lottery 6-49 --year all
python3 unified_lottery_scraper.py --lottery joker --year all

# Timp total: ~10-15 minute
# Rezultat: 3 fișiere JSON cu ~30 ani de date fiecare

# 2. Fă backup
tar -czf lottery_full_archive_2024-12-15.tar.gz *_data.json

# 3. Gata! Acum ai toate datele salvate PERMANENT
```

### Utilizare Ulterioară (Oricând)
```bash
# Folosești datele salvate de câte ori vrei:
python3 unified_pattern_finder.py --lottery 6-49 --input 6-49_data.json
python3 unified_pattern_finder.py --lottery joker --input joker_data.json

# NU mai trebuie să scrapuiești din nou!
```

### Update Periodic (Lunar/Trimestrial)
```bash
# La 3 luni, re-scrapuiește pentru update:
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Datele vechi sunt suprascrise cu versiunea nouă (include și extrageri noi)
```

## 📊 Dimensiuni Estimate Fișiere

| Loterie | Ani | Extrageri | Dimensiune JSON |
|---------|-----|-----------|----------------|
| 5/40 | 1 an | ~100 | ~35 KB |
| 5/40 | 30 ani | ~3000 | ~1 MB |
| 6/49 | 1 an | ~100 | ~35 KB |
| 6/49 | 30 ani | ~3000 | ~1 MB |
| Joker | 1 an | ~100 | ~75 KB (composite) |
| Joker | 30 ani | ~3000 | ~2.2 MB |
| **TOTAL (toate 3)** | **30 ani** | **~9000** | **~4.5 MB** |

**Concluzie**: Toate datele pentru 30 ani × 3 loterii = doar ~5 MB! 🎉

## 🚀 Exemple Concrete

### Exemplul 1: Primul Scraping (Complet)
```bash
cd /app/backend

# Scrapuiește tot ce există (1995-2025)
echo "Extragere Loto 6/49 - TOATE datele..."
python3 unified_lottery_scraper.py --lottery 6-49 --year all

echo "Extragere Joker - TOATE datele..."
python3 unified_lottery_scraper.py --lottery joker --year all

echo "Extragere Loto 5/40 - TOATE datele..."
python3 unified_lottery_scraper.py --lottery 5-40 --year all

echo "✅ GATA! Toate datele sunt salvate în:"
ls -lh *_data.json
```

**Output așteptat**:
```
5-40_data.json   1.1M
6-49_data.json   1.2M
joker_data.json  2.3M
```

### Exemplul 2: Scraping Rapid (Ultimii 5 Ani)
```bash
# Dacă vrei doar date recente pentru teste
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024,2023,2022,2021,2020
python3 unified_lottery_scraper.py --lottery joker --year 2024,2023,2022,2021,2020

# Mult mai rapid: ~1 minut total
```

### Exemplul 3: Update Lunar Automatizat
```bash
# Creează script de update
cat > /app/backend/monthly_update.sh << 'EOF'
#!/bin/bash
cd /app/backend

echo "📥 Monthly Update - $(date)"

# Update toate loteriile cu datele din anul curent
python3 unified_lottery_scraper.py --lottery 6-49 --year 2024
python3 unified_lottery_scraper.py --lottery joker --year 2024
python3 unified_lottery_scraper.py --lottery 5-40 --year 2024

echo "✅ Update complet!"
EOF

chmod +x /app/backend/monthly_update.sh

# Rulează manual când vrei:
./monthly_update.sh
```

## ⚠️ Note Importante

### 1. Rate Limiting
- Scraper-ul respectă serverul: `time.sleep(1)` între ani
- Pentru 30 ani = ~30 secunde doar pentru sleep
- E politicos față de noroc-chior.ro

### 2. Robustețe
- Gestionează automat erorile de rețea
- Continuă dacă un an eșuează
- Raportează progresul în timp real

### 3. Datele NU Expiră
- Fișierele JSON nu au "expirare"
- Poți folosi aceleași date ani de zile
- Update doar când vrei date noi

### 4. Portabilitate
- Copiază `*_data.json` pe alt PC
- Funcționează identic
- Nu depinde de baze de date

## 🎓 Best Practices

1. **Prima Dată**: Scrapuiește `--year all` pentru toate cele 3 loterii
2. **Backup**: Salvează fișierele JSON în safe location
3. **Update**: Re-scrapuiește doar când vrei date foarte recente
4. **Analiză**: Folosește aceleași fișiere JSON pentru multiple analize

## 📞 Troubleshooting

**Q: Cum văd ce am scrapuit deja?**
```bash
ls -lh *_data.json
python3 -c "import json; d=json.load(open('6-49_data.json')); print(f'{d[\"total_draws\"]} draws from {min(d[\"years\"])} to {max(d[\"years\"])}')"
```

**Q: Pot combina mai multe fișiere JSON?**
```bash
# Da, manual cu Python
python3 << EOF
import json
data1 = json.load(open('6-49_2023.json'))
data2 = json.load(open('6-49_2024.json'))
combined = data1['draws'] + data2['draws']
# Sort by date, remove duplicates, etc.
EOF
```

**Q: Se pot șterge datele accidental?**
- Nu, doar dacă ștergi manual fișierul JSON
- Recomandare: fă backup periodic

---

**Concluzie**: 
- ✅ Poți scrape oricâți ani dintr-o dată
- ✅ Datele rămân salvate PERMANENT în JSON
- ✅ Poți folosi datele de nelimitate ori
- ✅ `--year all` = simplu, rapid, complet! 🎉
