# Scripturi Seed Finding - Documentație

## ⚠️ DISCLAIMER IMPORTANT

Aceste scripturi sunt **EXPERIMENTE EDUCAȚIONALE** pentru a demonstra practic DE CE tehnicile de "seed finding" NU funcționează la loterii reale.

Rezultatele vor arăta că:
- Nu există seed-uri consistente în date
- "Potrivirile" sunt întâmplătoare
- Seed-urile NU pot prezice extrageri viitoare
- Datele NU provin dintr-un RNG cu seed

---

## 📦 Scripturi Disponibile

### 1. seed_finder.py - Căutare Seed-uri

Caută seed-uri RNG care recrează secvențe din istoricul de extrageri.

#### Utilizare:

```bash
# Caută seed pentru primele 2 extrageri
python3 seed_finder.py --input loto_data.json --draws 2

# Caută seed pentru 3 extrageri, testează 100k seeds
python3 seed_finder.py --input loto_data.json --draws 3 --seeds 100000

# Căutare progresivă prin tot istoricul
python3 seed_finder.py --input loto_data.json --draws 2 --progressive

# Cu Xorshift în loc de LCG
python3 seed_finder.py --input loto_data.json --draws 2 --rng xorshift
```

#### Ce face:
1. Testează mii/milioane de seed-uri random
2. Pentru fiecare seed, generează secvențe cu RNG (LCG sau Xorshift)
3. Compară cu extragerile reale
4. Găsește seed-urile cu cele mai bune "potriviri"
5. **Testează persistența** - arată că seed-urile NU funcționează pe termen lung

#### Output exemplu:
```
Găsite 652 seed-uri candidate:

1. Seed: 830,602
   Scor mediu: 50.00%
   
Test persistență:
- Extragerea 1: 3/6 match (50%)
- Extragerea 2: 3/6 match (50%)
- Extragerea 3: 0/6 match (0%)  ← scade rapid!
- Extragerea 4: 1/6 match (16%)
```

---

### 2. seed_evaluator.py - Evaluare Calitate Seed-uri

Evaluează "calitatea" seed-urilor găsite prin teste multiple.

#### Utilizare:

```bash
# Evaluează seed-uri specifice
python3 seed_evaluator.py --seeds 12345,67890,111213

# Găsește automat și evaluează top 5 seed-uri
python3 seed_evaluator.py --auto-find --top 5

# Cu Xorshift
python3 seed_evaluator.py --auto-find --top 10 --rng xorshift
```

#### Ce testează:
1. **Persistență** - câte extrageri consecutive "prezice"
2. **Consistență** - dacă seed funcționează în diferite părți ale istoricului
3. **Calitate match-uri** - distribuția potrivirilor
4. **Scor compozit** - evaluare overall

#### Output exemplu:
```
Evaluare seed: 830,602

1. Test Persistență (primele 20 extrageri):
   Scor mediu: 21.67%
   Match-uri medii: 1.3/6
   Persistență: 2 extrageri  ← MIC!

2. Test Consistență:
   Consistență (stdev): 0.156 (instabil)
   
3. Distribuție Match-uri:
   0/6: ████████ (8)
   1/6: ██████ (6)
   2/6: ████ (4)
   3/6: ██ (2)
```

---

### 3. seed_tracker.py - Urmărire Seed-uri în Timp

Urmărește performanța seed-urilor pe întreg istoricul și arată evoluția.

#### Utilizare:

```bash
# Testează 100k seed-uri, găsește cei mai "persistenți"
python3 seed_tracker.py --track 100000

# Analizează un seed specific
python3 seed_tracker.py --seed 830602

# Compară evoluția mai multor seed-uri
python3 seed_tracker.py --compare-evolution
```

#### Ce face:
1. Testează seed-uri pe ÎNTREG istoricul
2. Numără "hit-uri" (extrageri cu >=3 match-uri)
3. Calculează trend-uri (crește/scade performanța?)
4. Afișează grafice ASCII cu evoluție

#### Output exemplu:
```
Cei mai buni seed-uri:

Rank  Seed        Hits  Hit Rate  Avg Match
1     2,456,789   12    11.8%     1.42/6
2     8,234,567   11    10.8%     1.38/6

Evoluție seed 2,456,789:
2.50 | ▁▂▁▃▂▁▂▁▃▁▂▁▁▂
2.00 | ▃▄▃▅▄▃▄▃▅▃▄▃▃▄
1.50 | ████████████████
1.00 | ████████████████
      Start          End

Trend: descrescător  ← performanța SCADE!
```

---

## 🔬 Workflow Tipic de Experimentare

### Experiment 1: Căutare + Evaluare

```bash
# Pas 1: Extrage date
python3 loto_scraper.py --year 2024

# Pas 2: Caută seed-uri pentru 2 extrageri
python3 seed_finder.py --input loto_data.json --draws 2 --seeds 50000

# Pas 3: Notează cei mai buni seed (ex: 830602, 3105298)

# Pas 4: Evaluează-i detaliat
python3 seed_evaluator.py --seeds 830602,3105298
```

### Experiment 2: Tracking pe Termen Lung

```bash
# Testează 100k seed-uri pe tot istoricul
python3 seed_tracker.py --track 100000 --input loto_data.json

# Analizează cel mai bun seed găsit
python3 seed_tracker.py --seed [SEED_GASIT]
```

### Experiment 3: Căutare Progresivă

```bash
# Caută seed-uri pentru fiecare pereche de extrageri din istoric
python3 seed_finder.py --input loto_data.json --draws 2 --progressive

# Vezi că seed-urile diferă pentru fiecare secvență!
```

---

## 📊 Interpretarea Rezultatelor

### ✅ Ce VREI să vezi (pentru a confirma randomness):

1. **Scoruri scăzute** (~10-30%): Confirmare că nu e RNG
2. **Persistență mică** (1-3 extrageri): Seed-urile "mor" rapid
3. **Inconsistență ridicată**: Seed nu funcționează uniform în istoric
4. **Seed-uri diferite** pentru fiecare perioadă: Nu există "seed universal"
5. **Trend-uri instabile**: Performanța nu este predictibilă

### ❌ Ce NU ar trebui să vezi:

1. **Scoruri mari** (>70%): Ar indica RNG slab
2. **Persistență mare** (>10 extrageri): Ar indica pattern
3. **Consistență ridicată**: Ar indica seed real
4. **Același seed** funcționează peste tot: Imposibil pentru date aleatorii

---

## 💡 Exemple de Rezultate Așteptate

### Rezultat Tipic - Confirmare Randomness:

```
Seed: 830,602
Match-uri medii: 1.3/6 (21.67%)  ← ~Șansa random (16.67%)
Persistență: 2 extrageri          ← Scade rapid
Hit-uri: 5 din 102 (4.9%)        ← Puține

Concluzie: Nu există pattern, datele sunt aleatorii! ✓
```

### Ce AR însemna un seed "real" (nu se va întâmpla):

```
Seed: 123456 (IPOTETIC - NU REAL)
Match-uri medii: 5.2/6 (86.67%)  ← Prea mare!
Persistență: 95 extrageri         ← Prea mult!
Hit-uri: 89 din 102 (87.3%)      ← Imposibil!

Asta AR indica RNG - dar NU se va întâmpla la date reale!
```

---

## 🎓 Ce Învățăm din Aceste Experimente?

### 1. Diferența dintre RNG și Random True

**RNG (jocuri video):**
- Seed → Secvență predictibilă
- Persistență: ∞ (infinit, dacă știi seed-ul)
- Inversabil: DA

**Loterie Fizică:**
- Nu există seed
- Persistență: 0 (fiecare extragere independentă)
- Inversabil: NU

### 2. "Potrivirile" sunt Întâmplătoare

Când găsim un seed cu 3/6 match-uri:
- Șansa matematică random: C(6,3) × C(34,3) / C(40,6) ≈ 3.4%
- Ne așteptăm la ~3-4 astfel de "hit-uri" din 100
- Dacă găsim 5, e tot în limitele normalului statistic

### 3. Nu Există "Seed Universal"

Dacă datele ar fi dintr-un RNG:
- UN seed ar funcționa pentru TOT istoricul
- Vedem că avem nevoie de seed-uri DIFERITE pentru fiecare secvență
- Confirmare că NU e RNG

---

## 🔧 Parametri și Opțiuni

### seed_finder.py

```
--input FILE       Fișier JSON cu date (default: loto_data.json)
--draws N          Număr extrageri consecutive (2-5)
--seeds N          Număr seed-uri de testat (default: 100000)
--rng TYPE         Tip RNG: lcg sau xorshift (default: lcg)
--progressive      Căutare prin tot istoricul
```

### seed_evaluator.py

```
--input FILE       Fișier JSON cu date
--seeds LIST       Lista seed-uri: 12345,67890,111213
--auto-find        Găsește automat seed-uri buni
--top N            Top N seed-uri (cu --auto-find)
--rng TYPE         Tip RNG: lcg sau xorshift
```

### seed_tracker.py

```
--input FILE           Fișier JSON cu date
--track N              Testează N seed-uri random
--compare-evolution    Compară evoluția seed-urilor
--seed SEED            Analizează seed specific
--rng TYPE             Tip RNG: lcg sau xorshift
```

---

## ⏱️ Timp de Execuție

| Operație | Timp Estimat |
|----------|--------------|
| seed_finder (10k seeds, 2 draws) | 10-30 secunde |
| seed_finder (100k seeds, 2 draws) | 2-5 minute |
| seed_finder --progressive | 10-30 minute |
| seed_evaluator --auto-find --top 5 | 2-5 minute |
| seed_tracker --track 100000 | 30-60 minute |
| seed_tracker --seed SPECIFIC | 5-10 secunde |

---

## 🚨 Limitări și Observații

1. **Nu sunt optimizate pentru viteză maximă** - sunt tool-uri educaționale, nu production
2. **Folosesc RNG simple** (LCG, Xorshift) - loterii ar folosi RNG mult mai complicate (dacă ar folosi)
3. **Sample size** - testăm doar o fracțiune din spațiul de seed-uri posibil (2^32 = 4.3 miliarde)
4. **Rezultatele variază** - fiecare rulare testează seed-uri diferite (random sample)

---

## 📖 Resurse Suplimentare

- **seed_finder.py --help** - Help detaliat
- **seed_evaluator.py --help** - Help detaliat
- **seed_tracker.py --help** - Help detaliat
- **/app/backend/rng_demo.py** - Demo de bază RNG

---

## ✨ Concluzie

Aceste scripturi demonstrează PRACTIC, prin experimente, că:

✓ Datele de loterie NU provin dintr-un RNG cu seed
✓ "Potrivirile" sunt statistice normale, nu pattern-uri
✓ Nu există seed care să "prezică" consistent
✓ Tehnicile de seed finding din jocuri video NU se aplică la loterii

**Folosește aceste scripturi pentru a învăța, nu pentru a "găsi formula magică"!**

---

*Creat cu scop educațional - Decembrie 2025*
