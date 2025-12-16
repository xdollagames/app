#!/bin/bash
# Quick Ubuntu Setup Script for Lottery Analysis

set -e

echo "============================================================"
echo "  QUICK UBUNTU SETUP - Lottery Analysis System"
echo "============================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Please run as root or with sudo"
    exit 1
fi

echo "Step 1/6: Updating system..."
apt update && apt upgrade -y

echo ""
echo "Step 2/6: Installing essential packages..."
apt install -y \
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

echo ""
echo "Step 3/6: Creating project directory..."
mkdir -p /root/lottery-analysis
cd /root/lottery-analysis

echo ""
echo "Step 4/6: Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo ""
echo "Step 5/6: Installing Python dependencies..."
pip install \
    requests \
    beautifulsoup4 \
    numpy \
    scipy \
    lxml

echo ""
echo "Step 6/6: Creating directory structure..."
mkdir -p /root/lottery-analysis/backend
mkdir -p /root/lottery-analysis/data
mkdir -p /root/lottery-analysis/results

echo ""
echo "============================================================"
echo "✅ BASIC SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Copy your lottery analysis files to /root/lottery-analysis/backend/"
echo "  2. cd /root/lottery-analysis/backend"
echo "  3. source ../venv/bin/activate"
echo "  4. python3 unified_lottery_scraper.py --lottery 5-40 --year all"
echo "  5. ./progressive_test.sh"
echo ""
echo "For GPU setup (optional for future):"
echo "  sudo apt install -y nvidia-driver-550"
echo "  sudo reboot"
echo ""
