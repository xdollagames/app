# Sistem Unificat de Analiză Loterii

## 🎯 Prezentare Generală

Acest sistem unificat permite analiza oricărei loterii în mod configurabil. În loc de scripturi separate pentru fiecare loterie, avem un singur set de instrumente care funcționează pentru toate.

## 🎲 Loterii Suportate

1. **Loto 5/40** (`5-40`) - 6 numere din 1-40
2. **Loto 6/49** (`6-49`) - 6 numere din 1-49  
3. **Joker** (`joker`) - Format compus: 5 numere din 1-45 + 1 număr din 1-20

## 📁 Structura Sistemului

```
backend/
├── lottery_config.py              # Configurații pentru toate loteriile
├── unified_lottery_scraper.py     # Scraper unificat
├── unified_pattern_finder.py      # Analyzer unificat
├── advanced_rng_library.py        # Biblioteca RNG (neschimbată)
└── advanced_pattern_finder.py     # Pattern analyzer (neschimbat)
```

## 🚀 Utilizare

### Pasul 1: Extragere Date Istorice

```bash
# Pentru Loto 6/49 - ultimul an
python3 unified_lottery_scraper.py --lottery 6-49 --year 2025

# Pentru Joker - mai mulți ani
python3 unified_lottery_scraper.py --lottery joker --year 2024,2023,2022

# Pentru toate datele istorice disponibile
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Specificare output custom
python3 unified_lottery_scraper.py --lottery joker --year 2025 --output my_joker_data.json
```

**Output**: Fișier JSON cu datele istorice (ex: `6-49_data.json`, `joker_data.json`)

### Pasul 2: Analiză Pattern (Pragmatic Approach)

```bash
# Analiză pe Loto 6/49
python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input 6-49_data.json \
    --min-matches 3 \
    --success-threshold 0.65

# Analiză pe Joker (format compus)
python3 unified_pattern_finder.py \
    --lottery joker \
    --input joker_data.json \
    --min-matches 3 \
    --success-threshold 0.70

# Quick test (doar 4 RNG-uri rapide)
python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input 6-49_data.json \
    --quick-test

# Test specific RNG types
python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input 6-49_data.json \
    --rng-types lcg_weak xorshift_simple mersenne
```

**Output**: 
- Console: Progress și rezultate în timp real
- Fișier: `{lottery}_pragmatic_results.json` cu rezultate complete

### Parametri Disponibili

#### unified_lottery_scraper.py

| Parametru | Descriere | Default |
|-----------|-----------|---------|
| `--lottery` | Tipul de loterie (5-40, 6-49, joker) | **OBLIGATORIU** |
| `--year` | An sau ani (2025, 2024,2023, sau "all") | 2025 |
| `--output` | Fișier JSON de output | `{lottery}_data.json` |

#### unified_pattern_finder.py

| Parametru | Descriere | Default |
|-----------|-----------|---------|
| `--lottery` | Tipul de loterie | **OBLIGATORIU** |
| `--input` | Fișier JSON cu date | **OBLIGATORIU** |
| `--min-matches` | Minimum matches pentru success | 3 |
| `--success-threshold` | Success rate minim (0.0-1.0) | 0.65 |
| `--search-size` | Seeds de testat per extragere | 2,000,000 |
| `--seed-range` | Range pentru seeds | 0 10000000 |
| `--workers` | Număr de procese paralele | CPU count |
| `--rng-types` | RNG-uri specifice de testat | toate |
| `--quick-test` | Test rapid (doar 4 RNG-uri) | false |

## 📊 Exemple Complete

### Exemplul 1: Analiză Completă Loto 6/49

```bash
# 1. Extrage toate datele istorice
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# 2. Analiză pragmatică (3+ matches din 6)
python3 unified_pattern_finder.py \
    --lottery 6-49 \
    --input 6-49_data.json \
    --min-matches 3 \
    --success-threshold 0.70 \
    --workers 8

# 3. Vezi rezultatele
cat 6-49_pragmatic_results.json
```

### Exemplul 2: Quick Test pe Joker

```bash
# 1. Extrage ultimii 3 ani
python3 unified_lottery_scraper.py --lottery joker --year 2025,2024,2023

# 2. Quick test (doar RNG-uri rapide)
python3 unified_pattern_finder.py \
    --lottery joker \
    --input joker_data.json \
    --quick-test \
    --min-matches 3
```

### Exemplul 3: Test Specific RNG pe Loto 5/40

```bash
# Testează doar câteva RNG-uri specifice
python3 unified_pattern_finder.py \
    --lottery 5-40 \
    --input loto_data.json \
    --rng-types lcg_weak xorshift_simple lcg_glibc mersenne \
    --min-matches 4 \
    --success-threshold 0.75
```

## 🔍 Interpretarea Rezultatelor

### Success Rate >= 65-70%

**Semnificație**: Un RNG poate genera consistent matches bune
- ✅ RNG-ul se potrivește cu datele
- ✅ Dacă are și pattern în seeds → PREDICTIBIL
- ⚠️ ÎNSĂ: La loterii reale, acest lucru e EXTREM de improbabil

### Success Rate < 65%

**Semnificație**: RNG-ul nu se potrivește
- ❌ Seeds variază aleatoriu
- ❌ NU este acest tip de RNG

### NICIUN RNG nu atinge threshold

**Semnificație**: **CONFIRMARE că extragerea e fizică/aleatoare**
- ✓ Datele NU provin din niciun RNG software cunoscut
- ✓ Sistem impredictibil
- ✓ Confirmare loteriei reale

## 🎯 Format Output JSON

### Pentru Loterii Simple (5/40, 6/49)

```json
{
  "lottery_type": "6-49",
  "lottery_name": "Loto 6/49",
  "total_draws": 1247,
  "draws": [
    {
      "date": "2024-12-15",
      "date_str": "Du, 15 decembrie 2024",
      "numbers": [7, 23, 31, 38, 42, 45],
      "numbers_sorted": [7, 23, 31, 38, 42, 45],
      "year": 2024,
      "lottery_type": "6-49"
    }
  ]
}
```

### Pentru Loterii Compuse (Joker)

```json
{
  "lottery_type": "joker",
  "lottery_name": "Joker",
  "draws": [
    {
      "date": "2024-12-15",
      "numbers": [3, 12, 24, 35, 41, 8],
      "numbers_sorted": [3, 8, 12, 24, 35, 41],
      "composite_breakdown": {
        "part_1": {
          "numbers": [3, 12, 24, 35, 41],
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

## 🔧 Adăugarea Unei Noi Loterii

Pentru a adăuga o nouă loterie, editează `lottery_config.py`:

```python
LOTTERY_CONFIGS['noroc'] = LotteryConfig(
    name='Noroc',
    short_name='noroc',
    url_path='noroc',  # Path de pe site
    numbers_to_draw=6,
    min_number=0,
    max_number=999999,  # Ex: număr de 6 cifre
)
```

Apoi folosește:
```bash
python3 unified_lottery_scraper.py --lottery noroc --year 2025
python3 unified_pattern_finder.py --lottery noroc --input noroc_data.json
```

## ⚡ Performance Tips

1. **Quick Test First**: Folosește `--quick-test` pentru teste rapide
2. **Specific RNGs**: Dacă ai o suspiciune, testează doar RNG-uri specifice
3. **Workers**: Ajustează `--workers` în funcție de CPU-ul tău
4. **Search Size**: Reduce `--search-size` pentru teste mai rapide (dar mai puțin acurate)

## 📈 Diferențe față de Sistemul Vechi

| Aspect | Sistem Vechi | Sistem Nou Unificat |
|--------|-------------|---------------------|
| Scripturi | Separate pt fiecare loterie | Un singur set universal |
| Configurare | Hardcodat în cod | Configurabil prin parametri |
| Extensibilitate | Greu de extins | Ușor - doar adaugi config |
| Mentenanță | Modifici N fișiere | Modifici 1 fișier |
| Loterii noi | Copy-paste și modificare | Adaugi 5 linii în config |

## 🎓 Concepte Tehnice

### RNG Library (18 algoritmi)
- Testează toate tipurile majore de PRNG-uri
- De la simple (LCG) la complexe (Mersenne Twister, PCG)
- Include și algoritmi "slabi" notori (RANDU)

### Pragmatic Approach
- NU caută 100% match perfect
- Caută "good enough" - 3-4 din 6 numere (~50-70% success)
- Analizează pe termen LUNG (10-20 ani)
- Dacă găsește pattern în seeds → potential predictibil

### Composite Lotteries (Joker)
- Tratează fiecare componentă separat
- 5 numere din 1-45 folosind un RNG instance
- 1 număr din 1-20 folosind același RNG (state continuat)
- Analiză combinată pentru matches

## 📞 Troubleshooting

**Problem**: Scraper-ul nu găsește date
- **Soluție**: Verifică că site-ul noroc-chior.ro e accesibil și structura HTML e aceeași

**Problem**: Pattern finder e prea lent
- **Soluție**: Folosește `--quick-test` sau reduce `--search-size`

**Problem**: NICIUN RNG nu dă rezultate
- **Soluție**: Asta e normal! Înseamnă că loteria E reală și impredictibilă ✓

## 🎉 Concluzie

Sistemul unificat oferă:
- ✅ Flexibilitate maximă
- ✅ Cod curat și ușor de întreținut  
- ✅ Extensibilitate pentru orice loterie nouă
- ✅ Analiza pragmatică "good enough" pentru teste realiste
- ✅ Suport pentru formate compuse (Joker)

**Bucură-te de testare!** 🎲
