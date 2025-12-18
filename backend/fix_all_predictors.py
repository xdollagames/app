#!/usr/bin/env python3
"""
Script pentru a fixa TOATE predictorii - elimină comparațiile sortate
"""

import re
import os

def fix_predictor_file(filename):
    """Fix a single predictor file"""
    if not os.path.exists(filename):
        print(f"⚠️  {filename} nu există, sărim peste...")
        return False
    
    print(f"\n🔧 Procesez: {filename}")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    original_content = content
    changes_made = 0
    
    # 1. Înlocuiește target_sorted = sorted(numbers) cu target_exact = numbers
    pattern1 = r'target_sorted\s*=\s*sorted\(numbers\)'
    replacement1 = 'target_exact = numbers  # FIX: Ordinea EXACTĂ, nu sortată!'
    content, count1 = re.subn(pattern1, replacement1, content)
    changes_made += count1
    
    # 2. Înlocuiește target_sorted = sorted(target) cu target_exact = target
    pattern2 = r'target_sorted\s*=\s*sorted\(target\)'
    replacement2 = 'target_exact = target  # FIX: Ordinea EXACTĂ!'
    content, count2 = re.subn(pattern2, replacement2, content)
    changes_made += count2
    
    # 3. Înlocuiește sorted(generated) == target_sorted cu generated == target_exact
    pattern3 = r'sorted\(generated\)\s*==\s*target_sorted'
    replacement3 = 'generated == target_exact  # FIX: Compară ORDINEA!'
    content, count3 = re.subn(pattern3, replacement3, content)
    changes_made += count3
    
    # 4. Înlocuiește sorted(generated) == sorted(numbers) cu generated == numbers
    pattern4 = r'sorted\(generated\)\s*==\s*sorted\(numbers\)'
    replacement4 = 'generated == numbers  # FIX: Ordinea EXACTĂ!'
    content, count4 = re.subn(pattern4, replacement4, content)
    changes_made += count4
    
    # 5. Înlocuiește sorted(generated) == sorted(target) cu generated == target
    pattern5 = r'sorted\(generated\)\s*==\s*sorted\(target\)'
    replacement5 = 'generated == target  # FIX: Ordinea EXACTĂ!'
    content, count5 = re.subn(pattern5, replacement5, content)
    changes_made += count5
    
    # 6. Înlocuiește sorted(result) == target_sorted cu result == target_exact
    pattern6 = r'sorted\(result\)\s*==\s*target_sorted'
    replacement6 = 'result == target_exact  # FIX: Compară ORDINEA!'
    content, count6 = re.subn(pattern6, replacement6, content)
    changes_made += count6
    
    # 7. Înlocuiește utilizările rămase de target_sorted cu target_exact
    content = content.replace('target_sorted', 'target_exact')
    
    if content != original_content:
        # Salvează fișierul modificat
        backup_file = f"{filename}.backup"
        with open(backup_file, 'w') as f:
            f.write(original_content)
        print(f"   ✓ Backup salvat: {backup_file}")
        
        with open(filename, 'w') as f:
            f.write(content)
        print(f"   ✓ Modificat cu succes! ({changes_made} modificări)")
        return True
    else:
        print(f"   ℹ️  Nicio modificare necesară")
        return False

def main():
    print("=" * 80)
    print("🚀 FIX AUTOMAT - Elimină comparațiile sortate din TOATE predictorii")
    print("=" * 80)
    
    predictors = [
        'ultimate_predictor.py',
        'simple_predictor.py',
        'max_predictor.py',
        'gpu_predictor.py',
        'gpu_safe_predictor.py',
        'predict_xorshift.py',
    ]
    
    fixed_count = 0
    for predictor in predictors:
        if fix_predictor_file(predictor):
            fixed_count += 1
    
    print("\n" + "=" * 80)
    print(f"✅ Finalizat! {fixed_count}/{len(predictors)} fișiere modificate")
    print("=" * 80)
    
    print("\n⚠️  IMPORTANT: Șterge cache-ul vechi!")
    print("   rm seeds_cache.json")
    print("   echo '{}' > seeds_cache.json")
    
    print("\n📝 Verifică manual:")
    for predictor in predictors:
        print(f"   - {predictor}")

if __name__ == '__main__':
    main()
