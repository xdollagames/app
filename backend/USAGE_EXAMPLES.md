# Exemple de Utilizare - Sistem Loto 5/40

## 🚀 Start Rapid (Quick Start)

### Demo Rapid (5 minute)
```bash
cd /app/backend
bash demo_quick.sh
```
Acest script va:
1. Extrage datele pentru 2024
2. Efectua o analiză statistică
3. Demonstra conceptul de RNG reverse engineering
4. Genera câteva combinații de exemplu

---

## 📚 Exemple Detaliate

### 1. Extragere Date (loto_scraper.py)

#### Exemplu 1: Un singur an
```bash
python3 loto_scraper.py --year 2024
```
**Output:**
```
Extragere date pentru anul 2024...
  ✓ Extrase 102 extrageri pentru anul 2024
✓ Date salvate în: loto_data.json
  Total extrageri: 102
```

#### Exemplu 2: Ultimii 3 ani
```bash
python3 loto_scraper.py --year 2025,2024,2023
```

#### Exemplu 3: Toată arhiva (1995-2025)
```bash
python3 loto_scraper.py --year all
```
⚠️ **Atenție:** Această comandă va dura 5-10 minute și va face 30+ request-uri HTTP.

#### Exemplu 4: Specific fișier custom de output
```bash
python3 loto_scraper.py --year 2024 --output my_analysis.json
```

---

### 2. Analiză Statistică (loto_analyzer.py)

#### Exemplu 1: Analiză standard
```bash
python3 loto_analyzer.py --input loto_data.json
```

#### Exemplu 2: Top 15 rezultate
```bash
python3 loto_analyzer.py --input loto_data.json --top 15
```

#### Exemplu 3: Salvare output în fișier
```bash
python3 loto_analyzer.py --input loto_data.json > analysis_report.txt
```

**Output Fragment:**
```
======================================================================
ANALIZĂ STATISTICĂ LOTO 5/40
======================================================================
Total extrageri analizate: 102

----------------------------------------------------------------------
1. TOP 10 NUMERE CELE MAI FRECVENTE
----------------------------------------------------------------------
  10:   25 apariții (4.08%)
  24:   23 apariții (3.76%)
  18:   22 apariții (3.59%)
  ...

----------------------------------------------------------------------
2. TOP 10 PERECHI FRECVENTE
----------------------------------------------------------------------
   1-24:   8 apariții împreună
   4-18:   7 apariții împreună
  ...
```

---

### 3. Demonstrație RNG (rng_demo.py)

#### Exemplu 1: Demo complet
```bash
python3 rng_demo.py --demo
```

**Output Include:**
- Generare secvență cu seed cunoscut
- Inversarea pas cu pas a algoritmului
- Simulare seed finding
- Explicație detaliată de ce NU funcționează la loterii

#### Exemplu 2: Căutare seed specific
```bash
python3 rng_demo.py --find-seed 12345678
```

**Output:**
```
Căutare seed pentru output: 12345678...
✓ Găsit seed: 42198
```

---

### 4. Generator Combinații (predictor.py)

#### Exemplu 1: Strategie "frequency" (numere frecvente)
```bash
python3 predictor.py --strategy frequency
```

**Output:**
```
Combinatii generate cu strategia 'frequency':
--------------------------------------------------
  1.  2 - 10 - 15 - 18 - 24 - 27
```

#### Exemplu 2: Multiple combinații
```bash
python3 predictor.py --strategy balanced --count 10
```

#### Exemplu 3: Toate strategiile
```bash
python3 predictor.py --strategy all
```

**Output:**
```
Combinatii generate cu strategia 'frequency':
  1.  2 - 10 - 15 - 18 - 24 - 27

Combinatii generate cu strategia 'balanced':
  1.  3 -  8 - 12 - 21 - 29 - 35

Combinatii generate cu strategia 'hot':
  1.  6 - 10 - 14 - 16 - 22 - 24

Combinatii generate cu strategia 'cold':
  1.  5 - 13 - 17 - 31 - 34 - 38

Combinatii generate cu strategia 'mixed':
  1.  1 -  9 - 14 - 19 - 30 - 33

Combinatii generate cu strategia 'avoid_recent':
  1.  4 -  7 - 11 - 20 - 26 - 32

Combinatii generate cu strategia 'random':
  1.  2 -  8 - 15 - 23 - 28 - 37
```

#### Exemplu 4: Custom data file
```bash
python3 predictor.py --strategy hot --count 5 --data my_analysis.json
```

---

## 🧑‍💻 Workflow Tipic Complet

### Scenariul 1: Analiză rapidă pentru acest an
```bash
# Pas 1: Extrage date 2024
python3 loto_scraper.py --year 2024

# Pas 2: Analizează
python3 loto_analyzer.py --input loto_data.json --top 10

# Pas 3: Generează 5 combinații
python3 predictor.py --strategy mixed --count 5
```

### Scenariul 2: Analiză istorică completă
```bash
# Pas 1: Extrage toate datele (durează ~10 min)
python3 loto_scraper.py --year all --output loto_complete.json

# Pas 2: Analiză aprofundată
python3 loto_analyzer.py --input loto_complete.json --top 20

# Pas 3: Compară strategii
python3 predictor.py --strategy all --data loto_complete.json
```

### Scenariul 3: Experimentare educațională
```bash
# Înțelege cum funcționează RNG
python3 rng_demo.py --demo

# Testează seed finding
python3 rng_demo.py --find-seed 987654321

# Vezi diferite strategii
python3 predictor.py --strategy all
```

---

## 📊 Salvare și Export

### Salvare analiză în fișier text
```bash
python3 loto_analyzer.py --input loto_data.json --top 15 > raport_$(date +%Y%m%d).txt
```

### Generare combinații pentru săptămână
```bash
for strategy in frequency balanced hot cold mixed; do
    echo "=== Strategia: $strategy ===" >> combinatii_saptamana.txt
    python3 predictor.py --strategy $strategy --count 2 >> combinatii_saptamana.txt
    echo "" >> combinatii_saptamana.txt
done
```

### Backup date
```bash
cp loto_data.json loto_data_backup_$(date +%Y%m%d).json
```

---

## 🔍 Combinare cu Alte Tool-uri

### Cu `jq` pentru procesare JSON
```bash
# Număr total extrageri
jq '.total_draws' loto_data.json

# Extrage doar numerele din prima extragere
jq '.draws[0].numbers' loto_data.json

# Filtrează extrageri din 2024
jq '.draws[] | select(.year == 2024)' loto_data.json
```

### Cu `grep` pentru filtrare
```bash
# Găsește toate extragerile cu numărul 23
python3 loto_analyzer.py | grep "23:"

# Filtrează doar top rezultate
python3 loto_analyzer.py | grep "TOP"
```

---

## ⌛ Automatizare (Cron Jobs)

### Actualizare zilnică date
```bash
# Editează crontab
crontab -e

# Adaugă linia (rulează în fiecare zi la 10:00)
0 10 * * * cd /app/backend && python3 loto_scraper.py --year 2025
```

### Generare combinații săptămânale
```bash
# În fiecare luni la 08:00
0 8 * * 1 cd /app/backend && python3 predictor.py --strategy all > /home/user/combinatii_$(date +\%Y\%m\%d).txt
```

---

## 🐞 Troubleshooting - Exemple

### Problemă: "Module not found: bs4"
```bash
# Soluție
pip3 install beautifulsoup4 requests
```

### Problemă: "FileNotFoundError: loto_data.json"
```bash
# Soluție: Rulează mai întâi scraper-ul
python3 loto_scraper.py --year 2024
```

### Problemă: Scraper nu extrage date
```bash
# Verificare conexiune
ping -c 3 noroc-chior.ro

# Test manual URL
curl -I http://noroc-chior.ro/Loto/5-din-40/arhiva-rezultate.php?Y=2024
```

### Problemă: JSON invalid
```bash
# Validează JSON
python3 -m json.tool loto_data.json > /dev/null && echo "JSON valid" || echo "JSON invalid"

# Reextrage datele
rm loto_data.json
python3 loto_scraper.py --year 2024
```

---

## 🎯 Use Cases Avansate

### 1. Comparație strategii pe termen lung
```bash
#!/bin/bash
for i in {1..100}; do
    python3 predictor.py --strategy frequency --count 1 >> freq_results.txt
    python3 predictor.py --strategy random --count 1 >> random_results.txt
done

# Apoi analizează distribuția
```

### 2. Monitorizare numere "fierbinți" în timp
```bash
#!/bin/bash
echo "Evoluție numere fierbinți:" > hot_evolution.txt
for year in 2020 2021 2022 2023 2024; do
    echo "=== Anul $year ===" >> hot_evolution.txt
    python3 loto_scraper.py --year $year --output temp_$year.json
    python3 loto_analyzer.py --input temp_$year.json | grep "FIERBINȚI" -A 12 >> hot_evolution.txt
    rm temp_$year.json
done
```

### 3. Test statistici pattern par/impar
```bash
python3 loto_analyzer.py --input loto_data.json | \
    grep -A 10 "PAR/IMPAR" | \
    tee pattern_analysis.txt
```

---

## 📝 Template Script Personal

```bash
#!/bin/bash
# my_loto_routine.sh - Rutina mea personalizată Loto

DATE=$(date +%Y%m%d)
DIR="/app/backend"
OUTPUT_DIR="$HOME/loto_results"

mkdir -p "$OUTPUT_DIR"

echo "Actualizare date..."
python3 "$DIR/loto_scraper.py" --year 2025,2024 --output "$OUTPUT_DIR/data_$DATE.json"

echo "Analiză..."
python3 "$DIR/loto_analyzer.py" --input "$OUTPUT_DIR/data_$DATE.json" --top 10 > "$OUTPUT_DIR/analiza_$DATE.txt"

echo "Generare combinații (strategia mea preferată: mixed)..."
python3 "$DIR/predictor.py" --strategy mixed --count 5 --data "$OUTPUT_DIR/data_$DATE.json" > "$OUTPUT_DIR/combinatii_$DATE.txt"

echo "Gata! Rezultate în: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
```

**Utilizare:**
```bash
chmod +x my_loto_routine.sh
./my_loto_routine.sh
```

---

## ❓ Întrebări Frecvente (FAQ)

**Î: Câte combinații ar trebui să generez?**
R: Nu contează - toate au aceleași șanse! Generează câte îți face plăcere.

**Î: Care strategie este cea mai bună?**
R: NICIUNA! Toate au exact aceleași șanse de câștig. Sunt doar pentru experiment.

**Î: Pot folosi datele pentru predicție reală?**
R: NU. Extragerile sunt complet independente și aleatorii.

**Î: De ce există atunci toate aceste analize?**
R: Pentru educație, pentru a învăța despre statistică și pentru a demonstra că nu există "sistem magic".

**Î: Pot automatiza generarea de combinații?**
R: Da, dar nu va îmbunătăți șansele tale de câștig.

---

**Remember: Aceste tool-uri sunt pentru învățare și distracție, nu pentru "câștig garantat"!**

Probabilitatea la Loto 5/40 (Categoria I): **1 în 3.838.380** 🎲
