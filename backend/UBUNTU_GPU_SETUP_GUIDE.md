# 🚀 Ubuntu + RTX 5090 Setup Guide - Step by Step

## ⚠️ IMPORTANT: GPU Status

**Current Status**: Sistemul folosește **CPU multiprocessing** (nu GPU încă)
- ✅ Foarte rapid pe CPU multi-core
- ⚠️ GPU acceleration = stub existent, dar NU implementat complet
- 🎯 RTX 5090 va fi excelent pentru viitor, dar deocamdată CPU e suficient

**Totuși, hai să configurăm totul corect pentru viitor!**

---

## 📋 Prerequisites

- Ubuntu 22.04 / 24.04 LTS
- NVIDIA RTX 5090
- Root/sudo access
- Internet connection

---

## Step 1: Update System & Install Basics

```bash
# Connect via SSH
ssh root@your-server-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    python3 \
    python3-pip \
    python3-venv
```

**Time**: ~5-10 minutes

---

## Step 2: Install NVIDIA Drivers (For Future GPU Use)

```bash
# Check GPU
lspci | grep -i nvidia
# Should show: NVIDIA RTX 5090

# Add NVIDIA repository
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update

# Install latest NVIDIA driver
sudo apt install -y nvidia-driver-550

# Reboot
sudo reboot

# After reboot, verify
nvidia-smi
```

**Expected Output**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.xx.xx    Driver Version: 550.xx.xx    CUDA Version: 12.4  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 5090     Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   40C    P0    50W / 450W |      0MiB / 24576MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**Time**: ~10-15 minutes + reboot

---

## Step 3: Install CUDA Toolkit (Optional - For Future GPU Acceleration)

```bash
# Download CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

**Time**: ~15-20 minutes

---

## Step 4: Setup Project Directory

```bash
# Create project directory
mkdir -p /root/lottery-analysis
cd /root/lottery-analysis

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**Time**: ~2 minutes

---

## Step 5: Transfer Files From Current System

### Option A: Direct Download (If You Have Archive)

```bash
# If you have a .tar.gz backup
cd /root/lottery-analysis
wget https://your-server.com/lottery_system.tar.gz
tar -xzf lottery_system.tar.gz
```

### Option B: Copy From This System

```bash
# From THIS system (where we are now), create archive
cd /app
tar -czf lottery_system_complete.tar.gz backend/

# Then on Ubuntu server:
# Use scp to copy
scp lottery_system_complete.tar.gz root@your-server-ip:/root/lottery-analysis/
```

### Option C: Manual File Copy

**Copy these key files to Ubuntu**:
```
/app/backend/
├── lottery_config.py
├── advanced_rng_library.py
├── advanced_pattern_finder.py
├── unified_lottery_scraper.py
├── unified_pattern_finder.py
├── analyze_specific_year.sh
├── quick_analyze.sh
├── requirements.txt
└── *.md (all documentation)
```

**Time**: ~5-10 minutes

---

## Step 6: Install Python Dependencies

```bash
cd /root/lottery-analysis/backend
source ../venv/bin/activate

# Install requirements
pip install -r requirements.txt

# If requirements.txt missing, install manually:
pip install \
    requests \
    beautifulsoup4 \
    numpy \
    scipy \
    lxml

# Verify installation
python3 -c "
import requests
from bs4 import BeautifulSoup
import numpy as np
from scipy import stats
print('✅ All imports successful!')
"
```

**Time**: ~5-10 minutes

---

## Step 7: Test System

```bash
cd /root/lottery-analysis/backend

# Test 1: Verify RNG library
python3 -c "
from advanced_rng_library import RNG_TYPES
print(f'✅ RNG Library: {len(RNG_TYPES)} RNG-uri disponibile')
print('Lista:', list(RNG_TYPES.keys()))
"

# Test 2: Verify lottery config
python3 lottery_config.py

# Test 3: Check scraper help
python3 unified_lottery_scraper.py --help

# Test 4: Check pattern finder help
python3 unified_pattern_finder.py --help
```

**Expected Output**: No errors, all imports work

**Time**: ~2 minutes

---

## Step 8: Your Specific Workflow - Extract All Years for 5/40

```bash
cd /root/lottery-analysis/backend
source ../venv/bin/activate

# Extract ALL years for Loto 5/40
echo "Starting extraction - All years (1995-2025)"
date

python3 unified_lottery_scraper.py --lottery 5-40 --year all --output 5-40_ALL_DATA.json

date
echo "✅ Extraction complete!"

# Verify data
python3 -c "
import json
data = json.load(open('5-40_ALL_DATA.json'))
print(f'Total draws: {data[\"total_draws\"]}')
print(f'Years: {min(data[\"years\"])} - {max(data[\"years\"])}')
years_count = {}
for draw in data['draws']:
    year = draw['year']
    years_count[year] = years_count.get(year, 0) + 1
print('Draws per year:')
for year in sorted(years_count.keys()):
    print(f'  {year}: {years_count[year]} draws')
"
```

**Time**: ~3-5 minutes
**Output**: `5-40_ALL_DATA.json` with ~3000+ draws

---

## Step 9: Progressive Testing Script

Now let me create the EXACT workflow you requested!

```bash
# Create progressive testing script
cat > /root/lottery-analysis/backend/progressive_test.sh << 'SCRIPT_END'
#!/bin/bash
# Progressive Testing: Year by Year, then 2 years, then 3, etc.

set -e

cd /root/lottery-analysis/backend
source ../venv/bin/activate

LOTTERY="5-40"
ALL_DATA="${LOTTERY}_ALL_DATA.json"
RESULTS_DIR="progressive_results_$(date +%Y%m%d_%H%M%S)"

# Check if data exists
if [ ! -f "$ALL_DATA" ]; then
    echo "❌ Error: $ALL_DATA not found!"
    echo "Run extraction first:"
    echo "  python3 unified_lottery_scraper.py --lottery $LOTTERY --year all --output $ALL_DATA"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/progress.log"

echo "============================================================" | tee -a "$LOG_FILE"
echo "  PROGRESSIVE TESTING - Loto 5/40" | tee -a "$LOG_FILE"
echo "  Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Get all years from data
YEARS=$(python3 << 'EOF'
import json
data = json.load(open('5-40_ALL_DATA.json'))
years = sorted(set(d['year'] for d in data['draws']))
print(' '.join(map(str, years)))
EOF
)

YEARS_ARRAY=($YEARS)
TOTAL_YEARS=${#YEARS_ARRAY[@]}

echo "Total years available: $TOTAL_YEARS" | tee -a "$LOG_FILE"
echo "Years: ${YEARS_ARRAY[@]}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to extract and test specific years
test_years() {
    local years_to_test=$1
    local test_name=$2
    local data_file="${RESULTS_DIR}/${test_name}_data.json"
    local results_file="${RESULTS_DIR}/${test_name}_results.json"
    
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "TEST: $test_name" | tee -a "$LOG_FILE"
    echo "Years: $years_to_test" | tee -a "$LOG_FILE"
    echo "Started: $(date)" | tee -a "$LOG_FILE"
    
    # Extract data for specific years
    python3 << EOF
import json

with open('$ALL_DATA', 'r') as f:
    all_data = json.load(f)

years_list = [int(y) for y in '$years_to_test'.split(',')]
filtered_draws = [d for d in all_data['draws'] if d['year'] in years_list]

filtered_data = {
    'lottery_type': all_data['lottery_type'],
    'lottery_name': all_data['lottery_name'],
    'config': all_data['config'],
    'total_draws': len(filtered_draws),
    'years': years_list,
    'test_name': '$test_name',
    'draws': filtered_draws
}

with open('$data_file', 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"Extracted {len(filtered_draws)} draws for years: {years_list}")
EOF
    
    # Run analysis (quick test for speed)
    echo "Running analysis..." | tee -a "$LOG_FILE"
    python3 unified_pattern_finder.py \
        --lottery $LOTTERY \
        --input "$data_file" \
        --quick-test \
        --min-matches 3 \
        > "${RESULTS_DIR}/${test_name}_output.txt" 2>&1
    
    # Copy results
    if [ -f "${LOTTERY}_pragmatic_results.json" ]; then
        mv "${LOTTERY}_pragmatic_results.json" "$results_file"
    fi
    
    # Extract summary
    python3 << EOF
import json
try:
    with open('$results_file', 'r') as f:
        results = json.load(f)
    
    print("\nRESULTS SUMMARY:")
    if results.get('results'):
        for rng, data in results['results'].items():
            print(f"  {rng}: {data['success_rate']:.1%} ({data['success_count']}/{data['total_draws']})")
    else:
        print("  ❌ No patterns found (all < threshold)")
    
    if results.get('predictions'):
        print(f"\n  ✓ Predictions generated: {len(results['predictions'])}")
    else:
        print("\n  ✗ No predictions (no viable patterns)")
except:
    print("\n  ⚠ Results file not found or error")
EOF
    
    echo "Completed: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# ============================================================
# PHASE 1: Year by Year (up to 2010)
# ============================================================

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PHASE 1: Testing Year by Year (up to 2010)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for year in "${YEARS_ARRAY[@]}"; do
    if [ "$year" -le 2010 ]; then
        test_years "$year" "year_${year}"
    fi
done

# ============================================================
# PHASE 2: Testing with 2 years combined
# ============================================================

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PHASE 2: Testing with 2 Years Combined" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Start from 1995, combine 2 consecutive years
for ((i=0; i<$TOTAL_YEARS-1; i++)); do
    year1=${YEARS_ARRAY[$i]}
    year2=${YEARS_ARRAY[$i+1]}
    test_years "$year1,$year2" "combo_2years_${year1}_${year2}"
done

# ============================================================
# PHASE 3: Testing with 3 years combined
# ============================================================

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PHASE 3: Testing with 3 Years Combined" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for ((i=0; i<$TOTAL_YEARS-2; i++)); do
    year1=${YEARS_ARRAY[$i]}
    year2=${YEARS_ARRAY[$i+1]}
    year3=${YEARS_ARRAY[$i+2]}
    test_years "$year1,$year2,$year3" "combo_3years_${year1}_${year3}"
done

# ============================================================
# PHASE 4: Testing with increasing combinations
# ============================================================

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PHASE 4: Testing Progressive Combinations" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 5 years
for ((i=0; i<$TOTAL_YEARS-4; i++)); do
    years_combo=""
    for ((j=0; j<5; j++)); do
        if [ -z "$years_combo" ]; then
            years_combo="${YEARS_ARRAY[$i+$j]}"
        else
            years_combo="${years_combo},${YEARS_ARRAY[$i+$j]}"
        fi
    done
    year_start=${YEARS_ARRAY[$i]}
    year_end=${YEARS_ARRAY[$i+4]}
    test_years "$years_combo" "combo_5years_${year_start}_${year_end}"
done

# 10 years
for ((i=0; i<$TOTAL_YEARS-9; i++)); do
    years_combo=""
    for ((j=0; j<10; j++)); do
        if [ -z "$years_combo" ]; then
            years_combo="${YEARS_ARRAY[$i+$j]}"
        else
            years_combo="${years_combo},${YEARS_ARRAY[$i+$j]}"
        fi
    done
    year_start=${YEARS_ARRAY[$i]}
    year_end=${YEARS_ARRAY[$i+9]}
    test_years "$years_combo" "combo_10years_${year_start}_${year_end}"
done

# ============================================================
# FINAL: Test ALL years together
# ============================================================

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "FINAL PHASE: Testing ALL Years Together" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

all_years=$(IFS=,; echo "${YEARS_ARRAY[*]}")
test_years "$all_years" "ALL_YEARS_COMPLETE"

# ============================================================
# Generate Final Report
# ============================================================

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "GENERATING FINAL REPORT" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python3 << 'EOF'
import json
import os
from glob import glob

results_dir = [d for d in os.listdir('.') if d.startswith('progressive_results_')][0]
result_files = glob(f'{results_dir}/*_results.json')

print("\n" + "="*70)
print("FINAL SUMMARY REPORT")
print("="*70)

all_tests = []

for result_file in sorted(result_files):
    test_name = os.path.basename(result_file).replace('_results.json', '')
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        best_rate = 0
        best_rng = None
        
        if data.get('results'):
            for rng, info in data['results'].items():
                if info['success_rate'] > best_rate:
                    best_rate = info['success_rate']
                    best_rng = rng
        
        all_tests.append({
            'test': test_name,
            'best_rng': best_rng,
            'best_rate': best_rate,
            'has_predictions': bool(data.get('predictions'))
        })
    except:
        pass

# Sort by success rate
all_tests.sort(key=lambda x: x['best_rate'], reverse=True)

print(f"\nTotal tests performed: {len(all_tests)}")
print(f"\nTop 10 Results (by success rate):\n")

for i, test in enumerate(all_tests[:10], 1):
    status = "✓ PRED" if test['has_predictions'] else "✗"
    print(f"{i:2}. {test['test']:40s} | {test['best_rng'] or 'None':15s} | {test['best_rate']:.1%} | {status}")

print(f"\n\nTests with success rate >= 65%: {sum(1 for t in all_tests if t['best_rate'] >= 0.65)}")
print(f"Tests with predictions: {sum(1 for t in all_tests if t['has_predictions'])}")

print("\n" + "="*70)
EOF

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "ALL TESTING COMPLETE!" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "Results directory: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

SCRIPT_END

# Make executable
chmod +x /root/lottery-analysis/backend/progressive_test.sh
```

**Time**: ~2 minutes to create script

---

## Step 10: Run Your Progressive Test!

```bash
cd /root/lottery-analysis/backend
source ../venv/bin/activate

# Start in tmux (so it continues if you disconnect)
tmux new -s lottery

# Run progressive test
./progressive_test.sh

# To detach: Ctrl+B then D
# To reattach: tmux attach -t lottery
```

**Estimated Total Time**:
- Single year test: ~2-5 minutes
- 2 years combo: ~5-10 minutes each
- 3 years combo: ~10-15 minutes each
- 5 years: ~20-30 minutes each
- 10 years: ~45-60 minutes each
- ALL years: ~2-3 hours

**Total for all tests**: ~6-12 hours (depending on CPU power)

---

## Step 11: Monitor Progress

### In Another Terminal
```bash
# Watch progress
tail -f /root/lottery-analysis/backend/progressive_results_*/progress.log

# Check CPU usage
htop

# Check current test
ps aux | grep python
```

### Check Intermediate Results
```bash
cd /root/lottery-analysis/backend
ls progressive_results_*/

# View specific test results
cat progressive_results_*/year_2024_output.txt | tail -50
```

---

## 📊 Expected Results Structure

After completion, you'll have:

```
progressive_results_20241215_120000/
├── progress.log                          # Complete log
├── year_1995_data.json                   # Data for 1995
├── year_1995_results.json                # Results for 1995
├── year_1995_output.txt                  # Full output
├── year_1996_data.json                   # Data for 1996
├── ...
├── combo_2years_1995_1996_data.json      # 2 years combined
├── combo_2years_1995_1996_results.json
├── ...
├── combo_3years_1995_1997_data.json      # 3 years combined
├── ...
├── combo_5years_1995_1999_data.json      # 5 years combined
├── ...
├── combo_10years_1995_2004_data.json     # 10 years combined
├── ...
└── ALL_YEARS_COMPLETE_results.json       # Final test
```

---

## 🎯 Quick Reference Commands

### Start Fresh
```bash
cd /root/lottery-analysis/backend
source ../venv/bin/activate
./progressive_test.sh
```

### Resume (If Stopped)
```bash
# Progressive test script runs sequentially
# If stopped, edit script to skip completed tests
# OR just re-run - it will regenerate
```

### Check Results Anytime
```bash
cd /root/lottery-analysis/backend
ls progressive_results_*/

# Best tests summary
python3 << 'EOF'
import json
from glob import glob

results = glob('progressive_results_*/*_results.json')
for r in sorted(results):
    data = json.load(open(r))
    if data.get('results'):
        best = max(data['results'].values(), key=lambda x: x['success_rate'])
        print(f"{r}: {best['success_rate']:.1%}")
EOF
```

---

## ⚡ GPU Acceleration (Future)

Currently NOT used, but prepared:

```bash
# When GPU implementation is ready
# It will automatically detect and use GPU
nvidia-smi  # Monitor GPU usage during runs

# Expected speedup: 10-100x faster
# Current CPU time: ~2-3 hours for all
# Future GPU time: ~10-20 minutes for all
```

---

## 🔧 Troubleshooting

### Issue: Scraping fails
```bash
# Test connectivity
ping noroc-chior.ro

# Test manual scraping
curl http://noroc-chior.ro/Loto/5-din-40/arhiva-rezultate.php?Y=2024
```

### Issue: Out of memory
```bash
# Check memory
free -h

# If low, reduce search size
python3 unified_pattern_finder.py \
    --lottery 5-40 \
    --input data.json \
    --search-size 500000  # Reduce from 2M
```

### Issue: Too slow
```bash
# Increase workers
python3 unified_pattern_finder.py \
    --lottery 5-40 \
    --input data.json \
    --workers 32  # Use more CPU cores
```

---

## ✅ Final Checklist

Before starting:
- [ ] Ubuntu installed
- [ ] NVIDIA drivers working (`nvidia-smi` shows GPU)
- [ ] Python environment activated
- [ ] All files copied to `/root/lottery-analysis/backend`
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Test scripts working
- [ ] Data extracted (`5-40_ALL_DATA.json` exists)
- [ ] Progressive test script created and executable
- [ ] Running in `tmux` session

---

## 🎉 You're Ready!

```bash
cd /root/lottery-analysis/backend
source ../venv/bin/activate
tmux new -s lottery
./progressive_test.sh
```

**Enjoy your analysis! The RTX 5090 will be ready for future GPU acceleration!** 🚀
