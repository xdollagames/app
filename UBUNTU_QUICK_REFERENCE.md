# 🚀 Ubuntu RTX 5090 - Quick Reference Card

## 📋 Your Exact Workflow

### Phase 0: Initial Setup (One Time)

```bash
# 1. Connect to Ubuntu server
ssh root@YOUR_SERVER_IP

# 2. Run quick setup (if you have the script)
wget https://your-files/quick_ubuntu_setup.sh
chmod +x quick_ubuntu_setup.sh
sudo ./quick_ubuntu_setup.sh

# 3. Copy lottery analysis files
cd /root/lottery-analysis/backend
# Upload files here (all .py, .sh, requirements.txt)

# 4. Activate environment
source ../venv/bin/activate
```

**Time**: ~15-20 minutes

---

### Phase 1: Extract All Years for 5/40

```bash
cd /root/lottery-analysis/backend
source ../venv/bin/activate

# Extract ALL years (1995-2025)
python3 unified_lottery_scraper.py --lottery 5-40 --year all --output 5-40_ALL_DATA.json

# Verify
python3 -c "
import json
d = json.load(open('5-40_ALL_DATA.json'))
print(f'Total: {d[\"total_draws\"]} draws from {min(d[\"years\"])} to {max(d[\"years\"])}')
"
```

**Time**: ~3-5 minutes
**Result**: `5-40_ALL_DATA.json` (~3000 draws)

---

### Phase 2: Run Progressive Tests

```bash
cd /root/lottery-analysis/backend
source ../venv/bin/activate

# Start in tmux (safe from disconnection)
tmux new -s lottery

# Run progressive test
./progressive_test.sh

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t lottery
```

**What it does**:
1. ✅ Test each year individually (1995-2010)
2. ✅ Test 2 years combined
3. ✅ Test 3 years combined
4. ✅ Test 5 years combined
5. ✅ Test 10 years combined
6. ✅ Test ALL years together

**Total Time**: ~6-12 hours (CPU dependent)

---

## 🎯 Quick Commands Reference

### Check Progress
```bash
# View log
tail -f progressive_results_*/progress.log

# Count completed tests
ls progressive_results_*/*_results.json | wc -l

# Check current CPU usage
htop
```

### View Results Anytime
```bash
cd /root/lottery-analysis/backend

# List all result files
ls progressive_results_*/*_results.json

# Quick summary of best rates
python3 << 'EOF'
import json
from glob import glob

for f in sorted(glob('progressive_results_*/*_results.json')):
    data = json.load(open(f))
    if data.get('results'):
        best_rate = max(r['success_rate'] for r in data['results'].values())
        test_name = f.split('/')[-1].replace('_results.json', '')
        if best_rate >= 0.5:  # Show only interesting results
            print(f"{test_name:50s}: {best_rate:.1%}")
EOF
```

### Stop and Resume
```bash
# Stop: Ctrl+C in tmux

# Resume: Script needs to be modified to skip completed
# OR just let it re-run (will overwrite)

# Kill tmux session
tmux kill-session -t lottery
```

---

## 📊 Understanding Your Results

### What You're Looking For

**For REAL lottery (Noroc-chior.ro 5/40)**:
```
Expected Result:
  ❌ All RNG success rates: 20-30%
  ❌ No patterns found
  ✅ CONFIRMATION: Lottery is RANDOM (GOOD!)
```

**For FAKE/vulnerable lottery**:
```
Suspicious Result:
  ⚠️ One RNG: 70%+ success rate
  ⚠️ Pattern detected in seeds
  ⚠️ Predictions generated
  🔴 PROBLEM: Lottery might be predictable
```

### Result Files Explained

```
progressive_results_20241215_120000/
├── year_2024_results.json           # Single year analysis
│   └── Contains: success_rates, patterns, predictions
│
├── combo_2years_2023_2024_results.json  # 2 years combined
│   └── More data = better detection
│
├── combo_10years_2015_2024_results.json # 10 years
│   └── Long-term pattern analysis
│
└── ALL_YEARS_COMPLETE_results.json  # Ultimate test
    └── Maximum data, best detection
```

---

## ⚡ GPU Status

### Current: CPU Only
- Uses multiprocessing (all CPU cores)
- Very fast on powerful CPU
- RTX 5090: Ready for future, but not used yet

### Future: GPU Acceleration
```bash
# Will be implemented later
# Expected speedup: 10-100x
# Current all-years test: 2-3 hours
# Future with GPU: 10-20 minutes
```

### Check GPU (For Future)
```bash
nvidia-smi  # Should show RTX 5090

# Install CUDA (when needed)
sudo apt install -y nvidia-driver-550
sudo reboot
```

---

## 🔧 Troubleshooting

### Issue: Can't connect to server
```bash
# Check SSH
ssh -v root@YOUR_SERVER_IP

# Check firewall
sudo ufw status
```

### Issue: Scraping fails
```bash
# Test internet
ping google.com
ping noroc-chior.ro

# Test manual
curl http://noroc-chior.ro/Loto/5-din-40/arhiva-rezultate.php?Y=2024
```

### Issue: Out of memory
```bash
# Check RAM
free -h

# Reduce search size in script
# Edit progressive_test.sh, change:
--search-size 500000  # Instead of 2000000
```

### Issue: Too slow
```bash
# Check CPU usage
htop

# Increase workers
# Edit progressive_test.sh, add to python command:
--workers 32  # Or your CPU core count
```

### Issue: Script stuck
```bash
# Check what's running
ps aux | grep python

# Kill if needed
pkill -f unified_pattern_finder

# Restart
./progressive_test.sh
```

---

## 📁 File Transfer Options

### Option 1: SCP
```bash
# From your local machine
scp -r /app/backend/* root@YOUR_SERVER_IP:/root/lottery-analysis/backend/
```

### Option 2: Create Archive Here, Transfer
```bash
# On this system
cd /app
tar -czf lottery_complete.tar.gz backend/

# Transfer
scp lottery_complete.tar.gz root@YOUR_SERVER_IP:/root/lottery-analysis/

# On Ubuntu
cd /root/lottery-analysis
tar -xzf lottery_complete.tar.gz
```

### Option 3: Git (If You Have Repo)
```bash
# On Ubuntu
cd /root/lottery-analysis
git clone YOUR_REPO_URL backend/
```

---

## 📊 Expected Timeline

| Phase | Task | Time |
|-------|------|------|
| Setup | Ubuntu + packages | 15-20 min |
| Setup | Transfer files | 5-10 min |
| Setup | Install Python deps | 5-10 min |
| Data | Extract all years | 3-5 min |
| **Test** | **Single years (15 years)** | **30-75 min** |
| **Test** | **2-year combos (~30 tests)** | **2-5 hours** |
| **Test** | **3-year combos (~28 tests)** | **4-7 hours** |
| **Test** | **5-year combos (~25 tests)** | **8-12 hours** |
| **Test** | **10-year combos (~20 tests)** | **15-20 hours** |
| **Test** | **All years (1 test)** | **2-3 hours** |
| **TOTAL** | **Complete workflow** | **~30-50 hours** |

**With RTX 5090 (future GPU impl)**: ~3-5 hours total 🚀

---

## ✅ Success Checklist

Before starting progressive test:
- [ ] Ubuntu server running
- [ ] SSH access working
- [ ] Files copied to `/root/lottery-analysis/backend/`
- [ ] Python venv activated
- [ ] Dependencies installed
- [ ] `5-40_ALL_DATA.json` extracted
- [ ] `progressive_test.sh` created and executable
- [ ] Running in tmux session

---

## 🎯 Commands You'll Use Most

```bash
# Start everything
cd /root/lottery-analysis/backend
source ../venv/bin/activate
tmux new -s lottery
./progressive_test.sh

# Monitor (in another terminal)
tail -f progressive_results_*/progress.log

# Check results
ls progressive_results_*/*_results.json | wc -l

# View best results
python3 << 'EOF'
import json
from glob import glob
for f in sorted(glob('progressive_results_*/*_results.json')):
    data = json.load(open(f))
    if data.get('results'):
        best = max(r['success_rate'] for r in data['results'].values())
        if best >= 0.5:
            print(f"{f.split('/')[-1]:50s}: {best:.1%}")
EOF
```

---

## 🎉 You're All Set!

Your workflow:
1. ✅ Setup Ubuntu
2. ✅ Extract all years
3. ✅ Run progressive tests
4. ✅ Analyze results

**The RTX 5090 is ready for future GPU acceleration!**

**For detailed step-by-step, see**: `UBUNTU_GPU_SETUP_GUIDE.md`

🚀 **Good luck with your analysis!**
