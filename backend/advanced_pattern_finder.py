#!/usr/bin/env python3
"""
Advanced Pattern Finder - Toate formulele matematice posibile

Caută:
1. Liniar: S(n) = a*n + b
2. Pătratic: S(n) = a*n² + b*n + c
3. Cubic: S(n) = a*n³ + b*n² + c*n + d
4. Exponential: S(n) = a * b^n + c
5. Logaritmic: S(n) = a * log(n) + b
6. LCG chain: S(n+1) = (a*S(n) + c) mod m
7. Fibonacci: S(n) = S(n-1) + S(n-2)
8. Modular: S(n) = (a*n + b) mod m
9. Hash-based: S(n) = hash(n)
10. Timestamp correlation
11. Combinat: multiple patterns
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import statistics
from scipy.optimize import curve_fit
import hashlib


class AdvancedPatternFinder:
    def __init__(self, seeds: List[int], dates: List[str] = None):
        self.seeds = np.array(seeds, dtype=np.float64)
        self.n = np.arange(len(seeds), dtype=np.float64)
        self.dates = dates
        
    def fit_linear(self) -> Optional[Dict]:
        """S(n) = a*n + b"""
        try:
            coeffs = np.polyfit(self.n, self.seeds, 1)
            a, b = coeffs
            predicted = np.polyval(coeffs, self.n)
            r_squared = self._calculate_r_squared(self.seeds, predicted)
            
            if r_squared > 0.90:
                next_seed = int(a * len(self.seeds) + b)
                return {
                    'type': 'linear',
                    'formula': f'S(n) = {a:.2f}*n + {b:.2f}',
                    'r_squared': float(r_squared),
                    'next_seed': next_seed,
                    'coefficients': {'a': float(a), 'b': float(b)}
                }
        except:
            pass
        return None
    
    def fit_quadratic(self) -> Optional[Dict]:
        """S(n) = a*n² + b*n + c"""
        try:
            coeffs = np.polyfit(self.n, self.seeds, 2)
            a, b, c = coeffs
            predicted = np.polyval(coeffs, self.n)
            r_squared = self._calculate_r_squared(self.seeds, predicted)
            
            if r_squared > 0.90:
                next_n = len(self.seeds)
                next_seed = int(a * next_n**2 + b * next_n + c)
                return {
                    'type': 'quadratic',
                    'formula': f'S(n) = {a:.2e}*n² + {b:.2f}*n + {c:.2f}',
                    'r_squared': float(r_squared),
                    'next_seed': next_seed,
                    'coefficients': {'a': float(a), 'b': float(b), 'c': float(c)}
                }
        except:
            pass
        return None
    
    def fit_cubic(self) -> Optional[Dict]:
        """S(n) = a*n³ + b*n² + c*n + d"""
        try:
            if len(self.seeds) < 5:
                return None
            
            coeffs = np.polyfit(self.n, self.seeds, 3)
            a, b, c, d = coeffs
            predicted = np.polyval(coeffs, self.n)
            r_squared = self._calculate_r_squared(self.seeds, predicted)
            
            if r_squared > 0.90:
                next_n = len(self.seeds)
                next_seed = int(a * next_n**3 + b * next_n**2 + c * next_n + d)
                return {
                    'type': 'cubic',
                    'formula': f'S(n) = {a:.2e}*n³ + {b:.2e}*n² + {c:.2f}*n + {d:.2f}',
                    'r_squared': float(r_squared),
                    'next_seed': next_seed,
                    'coefficients': {'a': float(a), 'b': float(b), 'c': float(c), 'd': float(d)}
                }
        except:
            pass
        return None
    
    def fit_exponential(self) -> Optional[Dict]:
        """S(n) = a * b^n + c"""
        try:
            if len(self.seeds) < 3:
                return None
            
            def exp_func(x, a, b, c):
                return a * (b ** x) + c
            
            # Initial guess
            p0 = [self.seeds[0], 1.1, 0]
            
            params, _ = curve_fit(exp_func, self.n, self.seeds, p0=p0, maxfev=10000)
            a, b, c = params
            
            predicted = exp_func(self.n, a, b, c)
            r_squared = self._calculate_r_squared(self.seeds, predicted)
            
            if r_squared > 0.90 and abs(b - 1.0) > 0.01:  # Exclude b~1 (trivial)
                next_seed = int(exp_func(len(self.seeds), a, b, c))
                return {
                    'type': 'exponential',
                    'formula': f'S(n) = {a:.2f} * {b:.4f}^n + {c:.2f}',
                    'r_squared': float(r_squared),
                    'next_seed': next_seed,
                    'coefficients': {'a': float(a), 'b': float(b), 'c': float(c)}
                }
        except:
            pass
        return None
    
    def fit_logarithmic(self) -> Optional[Dict]:
        """S(n) = a * log(n+1) + b"""
        try:
            if len(self.seeds) < 3:
                return None
            
            log_n = np.log(self.n + 1)  # +1 to avoid log(0)
            coeffs = np.polyfit(log_n, self.seeds, 1)
            a, b = coeffs
            
            predicted = a * log_n + b
            r_squared = self._calculate_r_squared(self.seeds, predicted)
            
            if r_squared > 0.90:
                next_seed = int(a * np.log(len(self.seeds) + 1) + b)
                return {
                    'type': 'logarithmic',
                    'formula': f'S(n) = {a:.2f} * log(n+1) + {b:.2f}',
                    'r_squared': float(r_squared),
                    'next_seed': next_seed,
                    'coefficients': {'a': float(a), 'b': float(b)}
                }
        except:
            pass
        return None
    
    def fit_lcg_chain(self) -> Optional[Dict]:
        """S(n+1) = (a * S(n) + c) mod m"""
        if len(self.seeds) < 3:
            return None
        
        # Test common moduli
        moduli = [2**31, 2**32, 2**31 - 1, 10**9 + 7, 2**16]
        
        best_match = None
        best_score = 0
        
        for m in moduli:
            # Try to find a, c using first few equations
            # S1 = (a*S0 + c) mod m
            # S2 = (a*S1 + c) mod m
            # Solve for a, c
            
            seeds_int = [int(s) for s in self.seeds[:10]]
            
            for a in range(1, 10000, 50):
                for c in range(0, 10000, 50):
                    correct = 0
                    for i in range(len(seeds_int) - 1):
                        predicted = (a * seeds_int[i] + c) % m
                        if abs(predicted - seeds_int[i+1]) < m * 0.01:  # 1% tolerance
                            correct += 1
                    
                    score = correct / (len(seeds_int) - 1)
                    if score > best_score:
                        best_score = score
                        best_match = {'a': a, 'c': c, 'm': m}
        
        if best_match and best_score > 0.7:
            a, c, m = best_match['a'], best_match['c'], best_match['m']
            last_seed = int(self.seeds[-1])
            next_seed = (a * last_seed + c) % m
            
            return {
                'type': 'lcg_chain',
                'formula': f'S(n+1) = ({a}*S(n) + {c}) mod {m}',
                'r_squared': best_score,
                'next_seed': int(next_seed),
                'coefficients': {'a': a, 'c': c, 'm': m}
            }
        
        return None
    
    def fit_fibonacci(self) -> Optional[Dict]:
        """S(n) = a*S(n-1) + b*S(n-2)"""
        try:
            if len(self.seeds) < 5:
                return None
            
            # Test if Fibonacci-like
            matches = 0
            for i in range(2, len(self.seeds)):
                predicted = self.seeds[i-1] + self.seeds[i-2]
                error = abs(predicted - self.seeds[i]) / max(self.seeds[i], 1)
                if error < 0.1:  # 10% tolerance
                    matches += 1
            
            if matches > len(self.seeds) * 0.7:
                next_seed = int(self.seeds[-1] + self.seeds[-2])
                return {
                    'type': 'fibonacci',
                    'formula': 'S(n) = S(n-1) + S(n-2)',
                    'r_squared': matches / (len(self.seeds) - 2),
                    'next_seed': next_seed,
                    'coefficients': {}
                }
        except:
            pass
        return None
    
    def fit_modular(self) -> Optional[Dict]:
        """S(n) = (a*n + b) mod m"""
        moduli = [2**31, 2**32, 10**9 + 7, 10**6]
        
        for m in moduli:
            try:
                # Fit linear in modular space
                seeds_mod = self.seeds % m
                coeffs = np.polyfit(self.n, seeds_mod, 1)
                a, b = coeffs
                
                predicted = (a * self.n + b) % m
                matches = np.sum(np.abs(seeds_mod - predicted) < m * 0.01)
                
                if matches > len(self.seeds) * 0.8:
                    next_seed = int((a * len(self.seeds) + b) % m)
                    return {
                        'type': 'modular',
                        'formula': f'S(n) = ({a:.2f}*n + {b:.2f}) mod {m}',
                        'r_squared': matches / len(self.seeds),
                        'next_seed': next_seed,
                        'coefficients': {'a': float(a), 'b': float(b), 'm': m}
                    }
            except:
                continue
        
        return None
    
    def fit_hash_based(self) -> Optional[Dict]:
        """S(n) = hash(n) mod M"""
        # Test if seeds look like hash outputs
        try:
            # Check uniformity
            hist, _ = np.histogram(self.seeds, bins=10)
            uniformity = np.std(hist) / np.mean(hist)
            
            if uniformity < 0.3:  # Very uniform
                # Try to find hash pattern
                test_hashes = []
                for i in range(len(self.seeds)):
                    h = int(hashlib.md5(str(i).encode()).hexdigest(), 16)
                    test_hashes.append(h % 10000000)
                
                # Compare distribution
                return {
                    'type': 'hash_based',
                    'formula': 'S(n) ~ hash(n)',
                    'r_squared': 0.0,  # Cannot compute directly
                    'next_seed': int(hashlib.md5(str(len(self.seeds)).encode()).hexdigest(), 16) % 10000000,
                    'coefficients': {}
                }
        except:
            pass
        return None
    
    def _calculate_r_squared(self, actual, predicted):
        """Calculate R² score"""
        ss_tot = np.sum((actual - np.mean(actual))**2)
        ss_res = np.sum((actual - predicted)**2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def find_all_patterns(self) -> List[Dict]:
        """Run all pattern detection algorithms"""
        patterns = []
        
        print("Testing patterns...")
        
        methods = [
            ('Linear', self.fit_linear),
            ('Quadratic', self.fit_quadratic),
            ('Cubic', self.fit_cubic),
            ('Exponential', self.fit_exponential),
            ('Logarithmic', self.fit_logarithmic),
            ('LCG Chain', self.fit_lcg_chain),
            ('Fibonacci', self.fit_fibonacci),
            ('Modular', self.fit_modular),
            ('Hash-based', self.fit_hash_based),
        ]
        
        for name, method in methods:
            print(f"  Testing {name}...", end=' ')
            try:
                result = method()
                if result:
                    patterns.append(result)
                    print(f"✓ Found (R²={result['r_squared']:.3f})")
                else:
                    print("✗")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        return patterns


if __name__ == '__main__':
    # Test
    seeds = [100, 200, 300, 400, 500, 600]  # Linear
    finder = AdvancedPatternFinder(seeds)
    patterns = finder.find_all_patterns()
    
    print(f"\nFound {len(patterns)} patterns:")
    for p in patterns:
        print(f"  - {p['type']}: {p['formula']}")
