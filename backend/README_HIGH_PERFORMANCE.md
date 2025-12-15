# High Performance Seed Finding - Documentație

## 🚀 Overview

Aceste scripturi sunt optimizate pentru **calcule masive** pe hardware de înaltă performanță:
- **CPU multi-core** (64+ cores)
- **GPU CUDA** (NVIDIA)
- **Distributed clusters** (multiple mașini)

---

## 📦 Versiuni Disponibile

| Script | Hardware | Viteză | Use Case |
|--------|----------|--------|----------|
| **seed_finder_optimized.py** | CPU multi-core | ~100K-1M seeds/s | Servere CPU puternice |
| **seed_finder_gpu.py** | NVIDIA GPU | ~10M-100M seeds/s | Mașini cu GPU |
| **seed_finder_distributed.py** | Cluster | Scalabil linear | Data centers |

---

## 1️⃣ CPU Optimized - Pentru Servere Multi-Core

### Features:
✓ **Multiprocessing** - Folosește toate CPU cores
✓ **Checkpointing** - Salvează progres, poate fi întrerupt/reluat
✓ **Batch processing** - Procesare eficientă în batch-uri
✓ **Memory efficient** - Nu încarcă tot în RAM
✓ **Progress tracking** - ETA real-time
✓ **Incremental results** - Salvează rezultate pe parcurs

### Instalare:
```bash
# Numpy pentru calcule rapide
pip3 install numpy

# Optional: psutil pentru monitoring
pip3 install psutil
```

### Utilizare:

#### Exemplu 1: Test rapid (1 milion seeds)
```bash
python3 seed_finder_optimized.py \
    --seed-range 0 1000000 \
    --input loto_data.json \
    --draws 2 \
    --workers 8
```

#### Exemplu 2: Căutare masivă (1 miliard seeds)
```bash
python3 seed_finder_optimized.py \
    --seed-range 0 1000000000 \
    --input loto_data.json \
    --draws 2 \
    --workers 64 \
    --checkpoint checkpoint.json \
    --checkpoint-every 10000000
```

#### Exemplu 3: Resume din checkpoint
```bash
python3 seed_finder_optimized.py \
    --resume checkpoint.json \
    --workers 64
```

#### Exemplu 4: Full range (toate seed-urile 32-bit)
```bash
# AVERTISMENT: Va dura zile/săptămâni chiar pe hardware puternic!
python3 seed_finder_optimized.py \
    --seed-range 0 4294967296 \
    --input loto_data.json \
    --workers 128 \
    --checkpoint checkpoint_full.json \
    --checkpoint-every 100000000
```

### Parametri:

```
--input FILE              Fișier JSON cu date (default: loto_data.json)
--seed-range START END    Range de seeds (ex: 0 1000000000)
--draws N                 Număr extrageri consecutive (default: 2)
--workers N               Număr workers (default: toate cores)
--batch-size N            Seeds per batch (default: 10000)
--threshold FLOAT         Threshold minim scor (default: 0.25 = 25%)
--rng TYPE                LCG sau xorshift (default: lcg)
--checkpoint FILE         Fișier checkpoint
--checkpoint-every N      Seeds între checkpoints (default: 1000000)
--resume FILE             Resume din checkpoint
```

### Performanță Așteptată:

| CPU | Cores | Viteză Estimată |
|-----|-------|-----------------|
| AMD EPYC 7763 | 64 cores | ~500K-800K seeds/s |
| Intel Xeon Platinum | 56 cores | ~400K-600K seeds/s |
| AMD Ryzen 9 5950X | 16 cores | ~150K-250K seeds/s |
| Intel i9-12900K | 16 cores | ~120K-200K seeds/s |

**Timp pentru 1 miliard seeds:**
- 64 cores @ 600K/s: ~28 minute
- 16 cores @ 200K/s: ~83 minute (~1.4 ore)

---

## 2️⃣ GPU Version - Pentru NVIDIA CUDA

### Features:
✓ **CUDA acceleration** - Calcule paralele masive pe GPU
✓ **~10-100x mai rapid** decât CPU (depinde de GPU)
✓ **Batch processing** - Procesează milioane seeds simultan
✓ **Memory management** - Gestionare automată VRAM

### Cerințe:
```bash
# NVIDIA GPU cu CUDA support
# CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

# Numba cu CUDA
pip3 install numba

# Verificare CUDA
python3 -c "from numba import cuda; print('CUDA available:', cuda.is_available())"
```

### Utilizare:

#### Exemplu 1: Test GPU (10 milioane seeds)
```bash
python3 seed_finder_gpu.py \
    --seed-range 0 10000000 \
    --input loto_data.json \
    --draws 2 \
    --gpu-batch 1000000
```

#### Exemplu 2: Căutare masivă GPU (1 miliard)
```bash
python3 seed_finder_gpu.py \
    --seed-range 0 1000000000 \
    --input loto_data.json \
    --draws 2 \
    --gpu-batch 5000000 \
    --threshold 0.20
```

### Parametri:

```
--input FILE              Fișier JSON cu date
--seed-range START END    Range de seeds (REQUIRED)
--draws N                 Număr extrageri (default: 2)
--gpu-batch N             Seeds per GPU batch (default: 1000000)
--threshold FLOAT         Threshold minim (default: 0.25)
```

### Performanță GPU:

| GPU | VRAM | Viteză Estimată |
|-----|------|-----------------|
| NVIDIA H100 | 80GB | ~50M-100M seeds/s |
| NVIDIA A100 | 40GB | ~30M-60M seeds/s |
| NVIDIA V100 | 32GB | ~20M-40M seeds/s |
| RTX 4090 | 24GB | ~15M-30M seeds/s |
| RTX 3090 | 24GB | ~10M-20M seeds/s |
| RTX 3080 | 10GB | ~5M-10M seeds/s |

**Timp pentru 1 miliard seeds:**
- RTX 4090 @ 20M/s: **50 secunde!**
- RTX 3080 @ 7M/s: ~2.4 minute
- A100 @ 40M/s: **25 secunde!**

### VRAM Requirements:

- 1M seeds batch: ~50MB VRAM
- 5M seeds batch: ~250MB VRAM
- 10M seeds batch: ~500MB VRAM

**Recomandare:** Batch size = min(VRAM_GB * 1M, 10M)

---

## 3️⃣ Distributed Version - Pentru Cluster

### Features:
✓ **Multi-machine** - Distribuie pe N mașini
✓ **Master/Worker architecture**
✓ **Linear scaling** - 10 mașini = 10x viteză
✓ **Fault tolerant** - Workers pot intra/ieși dinamic
✓ **Network optimized** - Transfer minim de date

### Setup:

#### 1. Pregătește workers.txt pe master:
```
# workers.txt
192.168.1.101:64    # IP:NUM_CORES
192.168.1.102:64
192.168.1.103:32
192.168.1.104:16
```

#### 2. Start Master:
```bash
# Pe mașina master
python3 seed_finder_distributed.py \
    --mode master \
    --workers-file workers.txt \
    --seed-range 0 10000000000 \
    --input loto_data.json \
    --port 9999
```

#### 3. Start Workers (pe fiecare mașină worker):
```bash
# Pe mașina 192.168.1.101
python3 seed_finder_distributed.py \
    --mode worker \
    --master-ip 192.168.1.100 \
    --master-port 9999

# Repetă pe fiecare worker
```

### Performanță Cluster:

**Exemplu cluster:**
- 10x AMD EPYC 7763 (64 cores each)
- Total: 640 cores
- Viteză estimată: ~6M seeds/s

**Pentru 10 miliarde seeds:**
- 640 cores @ 6M/s: **~28 minute**

**Pentru 100 miliarde seeds:**
- 640 cores @ 6M/s: **~4.6 ore**

---

## 🔥 Maximizare Performanță

### CPU Optimization:

1. **Disable Hyper-Threading** dacă vrei predictibilitate
2. **CPU Affinity**: Pin workers la cores specifice
3. **Batch size tuning**: Testează 5K, 10K, 20K
4. **Threshold adjustment**: Threshold mai mare = mai rapid (dar mai puține rezultate)

```bash
# Exemple threshold
--threshold 0.20  # Relaxat - mai multe rezultate, mai lent
--threshold 0.30  # Strict - mai puține rezultate, mai rapid
--threshold 0.40  # Foarte strict - foarte rapid
```

### GPU Optimization:

1. **Batch size**: Maximizează fără să depășești VRAM
2. **GPU clock**: Overclock pentru +10-20% viteză
3. **Temperature**: Menține <80°C pentru throttling
4. **Multiple GPUs**: Rulează instanțe separate pe fiecare GPU

```bash
# Pentru multiple GPUs
CUDA_VISIBLE_DEVICES=0 python3 seed_finder_gpu.py --seed-range 0 500000000 &
CUDA_VISIBLE_DEVICES=1 python3 seed_finder_gpu.py --seed-range 500000000 1000000000 &
```

### Distributed Optimization:

1. **Network bandwidth**: 10Gbps+ recomandat
2. **Low latency**: <1ms între master-worker ideal
3. **Task granularity**: Chunk size 10M-100M seeds
4. **Load balancing**: Distribuie uniform pe workers

---

## 📊 Estimări Timp & Cost

### Scenarii Realiste:

#### Scenariu 1: Test exhaustiv moderat
- **Seeds:** 100 milioane
- **Hardware:** 1x RTX 3080
- **Timp:** ~3 minute
- **Cost:** $0.01 (cloud GPU @ $2/oră)

#### Scenariu 2: Căutare serioasă
- **Seeds:** 10 miliarde
- **Hardware:** 1x A100 GPU
- **Timp:** ~4 minute
- **Cost:** $0.20 (cloud GPU @ $3/oră)

#### Scenariu 3: Exhaustiv complet 32-bit
- **Seeds:** 4.3 miliarde (2^32)
- **Hardware:** 10x A100 GPUs
- **Timp:** ~20 minute
- **Cost:** ~$10 (10x GPU @ $3/oră)

#### Scenariu 4: Mega-exhaustiv
- **Seeds:** 1 trilion (pentru testare 2-3 draws)
- **Hardware:** Cluster 100x servers (6400 cores)
- **Timp:** ~2 zile
- **Cost:** ~$1000 (cloud compute)

### Cloud Providers - Cost Estimat:

| Provider | Instance | vCPUs | Price/hr | Seeds/s | $/Billion Seeds |
|----------|----------|-------|----------|---------|-----------------|
| AWS | c6a.48xlarge | 192 | $6.48 | ~1.5M | ~$1.20 |
| AWS | p3.16xlarge | 8x V100 | $24.48 | ~120M | ~$0.06 |
| GCP | c2-standard-60 | 60 | $3.20 | ~450K | ~$2.00 |
| Azure | HBv3 | 120 | $3.60 | ~900K | ~$1.10 |

**Recomandare:** GPU instances pentru seed finding - 10-20x mai cost-effective!

---

## 💾 Checkpoints & Resume

### Format Checkpoint:
```json
{
  "last_seed": 150000000,
  "results": [
    {"seed": 12345, "avg_score": 0.35, ...},
    ...
  ],
  "timestamp": 1234567890.123
}
```

### Best Practices:

1. **Checkpoint frequency:**
   - Slow network: Every 1M seeds
   - Fast compute: Every 10M seeds
   - Ultra-fast (GPU): Every 100M seeds

2. **Storage:**
   - Local SSD pentru speed
   - Cloud storage pentru backup
   - Sync periodic to cloud

3. **Recovery:**
   - Testează resume înainte de runs lungi
   - Păstrează multiple checkpoint versions

---

## 🧪 Testing & Validation

### Benchmark Scripts:

```bash
# Test CPU performance
time python3 seed_finder_optimized.py --seed-range 0 100000 --workers 4

# Test GPU performance
time python3 seed_finder_gpu.py --seed-range 0 1000000 --gpu-batch 100000

# Verifică scaling
# 1 worker
time python3 seed_finder_optimized.py --seed-range 0 100000 --workers 1
# 4 workers (ar trebui ~4x mai rapid)
time python3 seed_finder_optimized.py --seed-range 0 100000 --workers 4
```

### Validare Rezultate:

```python
# Verifică un seed găsit
from seed_finder_optimized import FastLCG

seed = 12345678
rng = FastLCG(seed)

# Target
target = [3, 4, 5, 7, 18, 28]

# Generated
generated = rng.generate_numbers(6, 1, 40)

# Matches
matches = len(set(generated) & set(target))
print(f"Matches: {matches}/6 ({matches/6:.1%})")
```

---

## 🚨 Limitări & Realitate

### Technical Limitations:

1. **Full 32-bit space:**
   - 4,294,967,296 seeds posibile
   - Chiar cu 100M seeds/s: ~43 secunde per draw
   - Pentru 3 draws: ~2 minute
   - Pentru 100 draws: ~72 minute

2. **64-bit impossibil:**
   - 18,446,744,073,709,551,616 seeds
   - La 100M seeds/s: **5,849 ANI**

3. **Multiple draws exponențial:**
   - 2 draws: feasible
   - 3 draws: challenging
   - 5 draws: extremely slow
   - 10+ draws: imposibil în practică

### Reality Check:

**CE VOI GĂSI:**
- Seeds cu 2-3 matches (33-50%)
- Persistență: 1-3 extrageri
- Inconsistență ridicată
- Seed-uri diferite pentru fiecare perioadă

**CE NU VOI GĂSI:**
- Seed "magic" cu 5-6 matches consistent
- Seed care funcționează >10 extrageri
- Seed "universal" pentru tot istoricul

**CONCLUZIE AȘTEPTATĂ:**
După ce vei testa miliarde/trilioni de seeds, vei demonstra EMPERIC că datele NU provin dintr-un RNG - confirmând că sunt extrageri fizice aleatorii!

---

## 📈 Monitoring & Debugging

### Progress Monitoring:

```bash
# Output real-time
python3 seed_finder_optimized.py ... 2>&1 | tee log.txt

# Monitor CPU
htop

# Monitor GPU
nvidia-smi -l 1

# Monitor network (distributed)
iftop -i eth0
```

### Common Issues:

**"Out of memory"**
- Reduce batch size
- Increase swap (not recommended for performance)
- Use distributed version

**"Slow performance"**
- Check CPU temperature (thermal throttling)
- Verify all cores being used: `htop`
- Test smaller range first

**"GPU not found"**
- Verify: `nvidia-smi`
- Check CUDA: `nvcc --version`
- Reinstall numba: `pip install --upgrade numba`

---

## 🎯 Workflow Complet

### Pentru Computație Masivă:

```bash
# 1. Test pe range mic
python3 seed_finder_optimized.py --seed-range 0 1000000 --workers 8

# 2. Verifică performanță
# → Notează seeds/s

# 3. Estimează timp pentru range mare
# 1 miliard seeds / (seeds/s) = secunde

# 4. Start căutare masivă cu checkpoint
python3 seed_finder_optimized.py \
    --seed-range 0 1000000000 \
    --workers 64 \
    --checkpoint big_run.json \
    --checkpoint-every 10000000

# 5. Monitorizează progres
tail -f log.txt

# 6. Dacă se întrerupe, resume
python3 seed_finder_optimized.py --resume big_run.json --workers 64

# 7. Analizează rezultate
python3 seed_evaluator.py --seeds SEED1,SEED2,SEED3
python3 seed_tracker.py --seed BEST_SEED
```

---

## 🔬 Experimentare Științifică

### Ipoteză:
"Dacă datele provin dintr-un RNG, vom găsi seed-uri cu persistență ridicată."

### Metodologie:
1. Testează N seeds (ex: 10 miliarde)
2. Găsește top seeds cu cel mai bun scor
3. Evaluează persistența acestor seeds
4. Analizează consistența în timp

### Rezultat Așteptat:
- Scoruri medii ~0.15-0.30 (aproape de șansa random 0.166)
- Persistență scăzută (1-3 extrageri)
- Seed-uri diferite pentru perioade diferite
- **Concluzie: NU există seed → datele sunt aleatorii**

---

## 📚 Referințe & Resurse

- [Numba CUDA Docs](https://numba.readthedocs.io/en/stable/cuda/)
- [Python Multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [LCG Parameters](https://en.wikipedia.org/wiki/Linear_congruential_generator)
- [Xorshift Algorithm](https://en.wikipedia.org/wiki/Xorshift)

---

## ✅ Checklist Înainte de Run Mare

- [ ] Testat pe range mic (1M seeds)
- [ ] Verificat performanță (seeds/s)
- [ ] Calculat timp estimat
- [ ] Setup checkpoint
- [ ] Verificat spațiu disc pentru rezultate
- [ ] Monitorizare setup (htop/nvidia-smi)
- [ ] Backup data file (loto_data.json)

---

**Succes cu experimentele! Când vei termina, vei avea dovada empirică solidă că loteriile NU au seed-uri!** 🚀

*Pentru mașinării de 100+ GPUs sau clustere enterprise, contactează-mă pentru optimizări custom.*
