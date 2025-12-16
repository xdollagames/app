# 🎲 Ghid Pentru TOATE 3 Loteriile - Super Simplu

## ✅ DA! Funcționează Exact La Fel!

Sistemul e **UNIFICAT** - doar schimbi un parametru:

```bash
--lottery 5-40   # Pentru Loto 5/40
--lottery 6-49   # Pentru Loto 6/49
--lottery joker  # Pentru Joker
```

**Asta e TOT ce trebuie schimbat!** 🎉

---

## 🎯 Setups Pentru Fiecare VPS

### VPS 1: Loto 5/40

```bash
# Conectare
ssh root@VPS1_IP

# Setup
apt update && apt install -y python3 python3-pip tmux
pip3 install requests beautifulsoup4 numpy scipy

# Folder
mkdir -p /root/loto540/backend
cd /root/loto540/backend
# (copiază fișierele aici)

# Ia datele
python3 unified_lottery_scraper.py --lottery 5-40 --year all

# Test
nano test.sh
```

**În test.sh pentru 5/40:**
```bash
#!/bin/bash
cd /root/loto540/backend

# Fiecare an
for year in {1995..2010}; do
  echo "Testing 5/40 year $year..."
  python3 -c "
import json
data = json.load(open('5-40_data.json'))
draws = [d for d in data['draws'] if d['year'] == $year]
new = dict(data); new['draws'] = draws; new['total_draws'] = len(draws)
json.dump(new, open('temp.json', 'w'))
  "
  python3 unified_pattern_finder.py --lottery 5-40 --input temp.json --quick-test > result_$year.txt 2>&1
done

# 2 ani
for year in {1995..2024}; do
  y2=$((year+1))
  echo "Testing 5/40: $year+$y2..."
  python3 -c "
import json
data = json.load(open('5-40_data.json'))
draws = [d for d in data['draws'] if d['year'] in [$year,$y2]]
new = dict(data); new['draws'] = draws; new['total_draws'] = len(draws)
json.dump(new, open('temp.json', 'w'))
  "
  python3 unified_pattern_finder.py --lottery 5-40 --input temp.json --quick-test > result_${year}_${y2}.txt 2>&1
done

echo "✅ 5/40 DONE!"
```

---

### VPS 2: Loto 6/49

```bash
# Conectare
ssh root@VPS2_IP

# Setup (la fel)
apt update && apt install -y python3 python3-pip tmux
pip3 install requests beautifulsoup4 numpy scipy

# Folder
mkdir -p /root/loto649/backend
cd /root/loto649/backend
# (copiază fișierele aici)

# Ia datele - DOAR schimbi 5-40 cu 6-49!
python3 unified_lottery_scraper.py --lottery 6-49 --year all

# Test
nano test.sh
```

**În test.sh pentru 6/49:**
```bash
#!/bin/bash
cd /root/loto649/backend

# Fiecare an
for year in {1995..2010}; do
  echo "Testing 6/49 year $year..."
  python3 -c "
import json
data = json.load(open('6-49_data.json'))
draws = [d for d in data['draws'] if d['year'] == $year]
new = dict(data); new['draws'] = draws; new['total_draws'] = len(draws)
json.dump(new, open('temp.json', 'w'))
  "
  python3 unified_pattern_finder.py --lottery 6-49 --input temp.json --quick-test > result_$year.txt 2>&1
done

# 2 ani
for year in {1995..2024}; do
  y2=$((year+1))
  echo "Testing 6/49: $year+$y2..."
  python3 -c "
import json
data = json.load(open('6-49_data.json'))
draws = [d for d in data['draws'] if d['year'] in [$year,$y2]]
new = dict(data); new['draws'] = draws; new['total_draws'] = len(draws)
json.dump(new, open('temp.json', 'w'))
  "
  python3 unified_pattern_finder.py --lottery 6-49 --input temp.json --quick-test > result_${year}_${y2}.txt 2>&1
done

echo "✅ 6/49 DONE!"
```

---

### VPS 3: Joker

```bash
# Conectare
ssh root@VPS3_IP

# Setup (la fel)
apt update && apt install -y python3 python3-pip tmux
pip3 install requests beautifulsoup4 numpy scipy

# Folder
mkdir -p /root/joker/backend
cd /root/joker/backend
# (copiază fișierele aici)

# Ia datele - DOAR schimbi cu joker!
python3 unified_lottery_scraper.py --lottery joker --year all

# Test
nano test.sh
```

**În test.sh pentru Joker:**
```bash
#!/bin/bash
cd /root/joker/backend

# Fiecare an
for year in {1995..2010}; do
  echo "Testing Joker year $year..."
  python3 -c "
import json
data = json.load(open('joker_data.json'))
draws = [d for d in data['draws'] if d['year'] == $year]
new = dict(data); new['draws'] = draws; new['total_draws'] = len(draws)
json.dump(new, open('temp.json', 'w'))
  "
  python3 unified_pattern_finder.py --lottery joker --input temp.json --quick-test > result_$year.txt 2>&1
done

# 2 ani
for year in {1995..2024}; do
  y2=$((year+1))
  echo "Testing Joker: $year+$y2..."
  python3 -c "
import json
data = json.load(open('joker_data.json'))
draws = [d for d in data['draws'] if d['year'] in [$year,$y2]]
new = dict(data); new['draws'] = draws; new['total_draws'] = len(draws)
json.dump(new, open('temp.json', 'w'))
  "
  python3 unified_pattern_finder.py --lottery joker --input temp.json --quick-test > result_${year}_${y2}.txt 2>&1
done

echo "✅ Joker DONE!"
```

---

## 📊 Diferențele FOARTE Mici

| Ce Schimbi | 5/40 | 6/49 | Joker |
|-----------|------|------|-------|
| **--lottery** | `5-40` | `6-49` | `joker` |
| **Fișier date** | `5-40_data.json` | `6-49_data.json` | `joker_data.json` |
| **Folder** | `/root/loto540` | `/root/loto649` | `/root/joker` |

**ATÂT! Restul e identic!** 🎯

---

## 🚀 Workflow Paralel Pe 3 VPS-uri

### VPS 1 (Loto 5/40)
```bash
ssh root@VPS1_IP
cd /root/loto540/backend
python3 unified_lottery_scraper.py --lottery 5-40 --year all
tmux new -s loto540
./test.sh
# Ctrl+B apoi D
```

### VPS 2 (Loto 6/49)
```bash
ssh root@VPS2_IP
cd /root/loto649/backend
python3 unified_lottery_scraper.py --lottery 6-49 --year all
tmux new -s loto649
./test.sh
# Ctrl+B apoi D
```

### VPS 3 (Joker)
```bash
ssh root@VPS3_IP
cd /root/joker/backend
python3 unified_lottery_scraper.py --lottery joker --year all
tmux new -s joker
./test.sh
# Ctrl+B apoi D
```

**Toate rulează în PARALEL pe fiecare VPS!** 🎉

---

## 📋 Checklist Pentru Fiecare VPS

### VPS 1 (5/40):
- [ ] Conectat: `ssh root@VPS1_IP`
- [ ] Python instalat
- [ ] Fișiere copiate în `/root/loto540/backend/`
- [ ] Date extrase: `5-40_data.json`
- [ ] Script `test.sh` creat
- [ ] Rulează în tmux

### VPS 2 (6/49):
- [ ] Conectat: `ssh root@VPS2_IP`
- [ ] Python instalat
- [ ] Fișiere copiate în `/root/loto649/backend/`
- [ ] Date extrase: `6-49_data.json`
- [ ] Script `test.sh` creat
- [ ] Rulează în tmux

### VPS 3 (Joker):
- [ ] Conectat: `ssh root@VPS3_IP`
- [ ] Python instalat
- [ ] Fișiere copiate în `/root/joker/backend/`
- [ ] Date extrase: `joker_data.json`
- [ ] Script `test.sh` creat
- [ ] Rulează în tmux

---

## 🎯 Copy-Paste Ready Commands

### Pentru 5/40 (VPS 1):
```bash
ssh root@VPS1_IP
apt update && apt install -y python3 python3-pip tmux
pip3 install requests beautifulsoup4 numpy scipy
mkdir -p /root/loto540/backend && cd /root/loto540/backend
# Upload fișiere aici
python3 unified_lottery_scraper.py --lottery 5-40 --year all
# Creează test.sh (vezi mai sus)
chmod +x test.sh
tmux new -s loto540
./test.sh
```

### Pentru 6/49 (VPS 2):
```bash
ssh root@VPS2_IP
apt update && apt install -y python3 python3-pip tmux
pip3 install requests beautifulsoup4 numpy scipy
mkdir -p /root/loto649/backend && cd /root/loto649/backend
# Upload fișiere aici
python3 unified_lottery_scraper.py --lottery 6-49 --year all
# Creează test.sh (vezi mai sus)
chmod +x test.sh
tmux new -s loto649
./test.sh
```

### Pentru Joker (VPS 3):
```bash
ssh root@VPS3_IP
apt update && apt install -y python3 python3-pip tmux
pip3 install requests beautifulsoup4 numpy scipy
mkdir -p /root/joker/backend && cd /root/joker/backend
# Upload fișiere aici
python3 unified_lottery_scraper.py --lottery joker --year all
# Creează test.sh (vezi mai sus)
chmod +x test.sh
tmux new -s joker
./test.sh
```

---

## 🔍 Monitor Toate 3 VPS-urile

**Terminal 1 (VPS1 - 5/40):**
```bash
ssh root@VPS1_IP
tmux attach -t loto540
```

**Terminal 2 (VPS2 - 6/49):**
```bash
ssh root@VPS2_IP
tmux attach -t loto649
```

**Terminal 3 (VPS3 - Joker):**
```bash
ssh root@VPS3_IP
tmux attach -t joker
```

---

## 📊 Colectează Rezultatele

După ce toate se termină, copiază rezultatele:

```bash
# De pe VPS1
scp root@VPS1_IP:/root/loto540/backend/result_*.txt ./results_540/

# De pe VPS2
scp root@VPS2_IP:/root/loto649/backend/result_*.txt ./results_649/

# De pe VPS3
scp root@VPS3_IP:/root/joker/backend/result_*.txt ./results_joker/
```

---

## 🎓 Comparație Finală

```bash
# Pe computerul tău local
echo "=== COMPARAȚIE 5/40 vs 6/49 vs Joker ==="

echo "5/40 - Top results:"
grep "success rate" results_540/*.txt | grep -v "0.0%" | sort -t: -k2 -rn | head -5

echo ""
echo "6/49 - Top results:"
grep "success rate" results_649/*.txt | grep -v "0.0%" | sort -t: -k2 -rn | head -5

echo ""
echo "Joker - Top results:"
grep "success rate" results_joker/*.txt | grep -v "0.0%" | sort -t: -k2 -rn | head -5
```

---

## ⏱️ Timeline

| VPS | Loterie | Timp Estimat |
|-----|---------|--------------|
| VPS1 | 5/40 | 10-20 ore |
| VPS2 | 6/49 | 10-20 ore |
| VPS3 | Joker | 10-20 ore |

**În PARALEL: Toate 3 se termină în ~20 ore!** 🚀

---

## 💡 Tips

### Economisești Timp:
- Toate 3 rulează SIMULTAN pe VPS-uri diferite
- Nu aștepți 60 ore - doar 20 ore!

### Economisești Bani:
- Poți folosi VPS-uri ieftine ($5-10/lună)
- După ce se termină, închizi VPS-urile

### Organizare:
- Fiecare loterie pe VPS separat = curat
- Rezultate separate = ușor de comparat

---

## ✅ Rezumat Ultra Simplu

**Pentru fiecare VPS, doar schimbi 3 lucruri:**

1. **Numele loteriei în comandă:**
   - `--lottery 5-40`
   - `--lottery 6-49`
   - `--lottery joker`

2. **Numele fișierului date:**
   - `5-40_data.json`
   - `6-49_data.json`
   - `joker_data.json`

3. **Folderul:**
   - `/root/loto540/`
   - `/root/loto649/`
   - `/root/joker/`

**RESTUL E 100% IDENTIC!** 🎯

---

## 🎉 Concluzie

**DA! Funcționează EXACT LA FEL pentru toate 3!**

- ✅ Același cod
- ✅ Aceleași scripturi
- ✅ Doar schimbi `--lottery 5-40/6-49/joker`
- ✅ Poți rula pe 3 VPS-uri în paralel
- ✅ Super simplu!

**Sistemul e UNIFICAT - acesta e scopul!** 🚀
