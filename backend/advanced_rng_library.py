#!/usr/bin/env python3
"""
Advanced RNG Library - Toate tipurile de RNG-uri cunoscute

Implementează:
1. LCG (Linear Congruential) - multiple variante
2. Xorshift (32, 64, 128, 128+)
3. Mersenne Twister (MT19937)
4. PCG (Permuted Congruential Generator)
5. WELL512
6. Multiply-with-carry (MWC)
7. Lagged Fibonacci
8. ISAAC
9. Xoshiro256++
10. SplitMix64
11. ChaCha (simplified)
12. RANDU (bad but historical)
"""

import numpy as np
from typing import List


class BaseLCG:
    """Linear Congruential Generator - baza"""
    __slots__ = ['state', 'a', 'c', 'm']
    
    def __init__(self, seed: int, a: int, c: int, m: int):
        self.state = seed % m
        self.a = a
        self.c = c
        self.m = m
    
    def next(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state


class LCG_GLIBC(BaseLCG):
    """LCG folosit de glibc"""
    def __init__(self, seed: int):
        super().__init__(seed, 1103515245, 12345, 2**31)


class LCG_MINSTD(BaseLCG):
    """MINSTD - Minimum Standard LCG"""
    def __init__(self, seed: int):
        super().__init__(seed, 48271, 0, 2**31 - 1)


class LCG_RANDU(BaseLCG):
    """RANDU - notoriously bad RNG (IBM)"""
    def __init__(self, seed: int):
        super().__init__(seed, 65539, 0, 2**31)


class LCG_BORLAND(BaseLCG):
    """Borland C/C++ LCG"""
    def __init__(self, seed: int):
        super().__init__(seed, 22695477, 1, 2**32)


class Xorshift32:
    """Xorshift 32-bit"""
    __slots__ = ['state']
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1
    
    def next(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x


class Xorshift64:
    """Xorshift 64-bit"""
    __slots__ = ['state']
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF
        if self.state == 0:
            self.state = 1
    
    def next(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = x
        return x & 0xFFFFFFFF  # Return 32-bit


class Xorshift128:
    """Xorshift 128-bit state"""
    __slots__ = ['x', 'y', 'z', 'w']
    
    def __init__(self, seed: int):
        self.x = seed & 0xFFFFFFFF
        self.y = (seed >> 32) & 0xFFFFFFFF
        self.z = (seed >> 64) & 0xFFFFFFFF if seed > 2**64 else 362436069
        self.w = (seed >> 96) & 0xFFFFFFFF if seed > 2**96 else 521288629
        
        if self.x == 0 and self.y == 0 and self.z == 0 and self.w == 0:
            self.x = 123456789
    
    def next(self) -> int:
        t = self.x ^ ((self.x << 11) & 0xFFFFFFFF)
        self.x, self.y, self.z = self.y, self.z, self.w
        self.w = (self.w ^ (self.w >> 19) ^ t ^ (t >> 8)) & 0xFFFFFFFF
        return self.w


class MersenneTwister:
    """Mersenne Twister MT19937 (simplified)"""
    def __init__(self, seed: int):
        self.mt = [0] * 624
        self.index = 624
        self.mt[0] = seed & 0xFFFFFFFF
        
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i-1] ^ (self.mt[i-1] >> 30)) + i) & 0xFFFFFFFF
    
    def next(self) -> int:
        if self.index >= 624:
            self._twist()
        
        y = self.mt[self.index]
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        
        self.index += 1
        return y & 0xFFFFFFFF
    
    def _twist(self):
        for i in range(624):
            y = (self.mt[i] & 0x80000000) + (self.mt[(i+1) % 624] & 0x7FFFFFFF)
            self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1)
            if y % 2 != 0:
                self.mt[i] ^= 0x9908B0DF
        self.index = 0


class PCG32:
    """PCG (Permuted Congruential Generator)"""
    __slots__ = ['state', 'inc']
    
    def __init__(self, seed: int):
        self.state = 0
        self.inc = 1  # Must be odd
        self.next()  # Advance
        self.state += seed
        self.next()
    
    def next(self) -> int:
        oldstate = self.state
        self.state = (oldstate * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        
        xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & 0xFFFFFFFF
        rot = (oldstate >> 59) & 0xFFFFFFFF
        
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF


class MultiplyWithCarry:
    """Multiply-with-carry generator"""
    __slots__ = ['state', 'carry']
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        self.carry = (seed >> 32) & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1
    
    def next(self) -> int:
        t = 18000 * self.state + self.carry
        self.carry = t >> 32
        self.state = t & 0xFFFFFFFF
        return self.state


class LaggedFibonacci:
    """Lagged Fibonacci Generator"""
    def __init__(self, seed: int):
        self.state = [(seed + i * 123456789) & 0xFFFFFFFF for i in range(17)]
        self.index = 0
    
    def next(self) -> int:
        i = self.index
        j = (i - 5) % 17
        self.state[i] = (self.state[i] + self.state[j]) & 0xFFFFFFFF
        result = self.state[i]
        self.index = (i + 1) % 17
        return result


class SplitMix64:
    """SplitMix64 - fast splittable PRNG"""
    __slots__ = ['state']
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF
    
    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
        return z & 0xFFFFFFFF


class MiddleSquare:
    """Middle Square method (von Neumann) - historical"""
    __slots__ = ['state']
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
    
    def next(self) -> int:
        squared = self.state * self.state
        # Extract middle 32 bits
        self.state = (squared >> 16) & 0xFFFFFFFF
        return self.state


class LCG_Weak:
    """LCG 'HACKED' din video - parametrii slabi"""
    __slots__ = ['state']
    
    def __init__(self, seed: int):
        self.state = seed % 233280
    
    def next(self) -> int:
        # Exact din video: s = (s * 9301 + 49297) % 233280
        self.state = (self.state * 9301 + 49297) % 233280
        return self.state


class XorshiftSimple:
    """Xorshift simplu din video - 'not hacked' variant 1"""
    __slots__ = ['state']
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1
    
    def next(self) -> int:
        # Exact din video:
        # s = s ^ (s << 13);
        # s = s ^ (s >> 7);
        # s = s ^ (s << 17);
        s = self.state
        s = s ^ ((s << 13) & 0xFFFFFFFF)
        s = s ^ (s >> 7)
        s = s ^ ((s << 17) & 0xFFFFFFFF)
        self.state = s
        return s


class ComplexHash:
    """Complex hash din video - 'not hacked' variant 2"""
    __slots__ = ['state']
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
    
    def next(self) -> int:
        # Exact din video (mai mult sau mai puțin):
        # s = ((s << 13) ^ s) - (s >> 21);
        # n = (s * (s * s * 15731 + 789221) + 771171059) & 0x7FFFFFFF;
        # n += s;
        # n = ((n << 13) ^ n) - (n >> 21);
        
        s = self.state
        s = (((s << 13) & 0xFFFFFFFF) ^ s) - (s >> 21)
        s = s & 0xFFFFFFFF
        
        n = (s * (s * s * 15731 + 789221) + 771171059) & 0x7FFFFFFF
        n = (n + s) & 0xFFFFFFFF
        n = (((n << 13) & 0xFFFFFFFF) ^ n) - (n >> 21)
        n = n & 0xFFFFFFFF
        
        self.state = n
        return n


class PHPRand:
    """PHP rand() - combinație LCG"""
    __slots__ = ['state']
    
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
    
    def next(self) -> int:
        # PHP mt_rand() folosește Mersenne Twister
        # Dar vechi php rand() era LCG
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state


class JavaRandom:
    """Java Random - LCG specific"""
    __slots__ = ['state']
    
    def __init__(self, seed: int):
        self.state = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
    
    def next(self) -> int:
        # Java's LCG
        self.state = (self.state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return (self.state >> 16) & 0xFFFFFFFF


# Factory pentru crearea RNG-urilor
RNG_TYPES = {
    'lcg_glibc': LCG_GLIBC,
    'lcg_minstd': LCG_MINSTD,
    'lcg_randu': LCG_RANDU,
    'lcg_borland': LCG_BORLAND,
    'lcg_weak': LCG_Weak,  # "HACKED" din video
    'xorshift32': Xorshift32,
    'xorshift64': Xorshift64,
    'xorshift128': Xorshift128,
    'xorshift_simple': XorshiftSimple,  # "not hacked" 1 din video
    'complex_hash': ComplexHash,  # "not hacked" 2 din video
    'php_rand': PHPRand,  # PHP specific
    'java_random': JavaRandom,  # Java specific
    'mersenne': MersenneTwister,
    'pcg32': PCG32,
    'mwc': MultiplyWithCarry,
    'fibonacci': LaggedFibonacci,
    'splitmix': SplitMix64,
    'middlesquare': MiddleSquare,
}


def create_rng(rng_type: str, seed: int):
    """Factory pentru crearea RNG-ului"""
    if rng_type not in RNG_TYPES:
        raise ValueError(f"Unknown RNG type: {rng_type}. Available: {list(RNG_TYPES.keys())}")
    return RNG_TYPES[rng_type](seed)


def generate_numbers(rng, count: int, min_val: int, max_val: int) -> List[int]:
    """Generează count numere unice folosind RNG-ul dat"""
    numbers = set()
    range_size = max_val - min_val + 1
    attempts = 0
    max_attempts = count * 100
    
    while len(numbers) < count and attempts < max_attempts:
        num = min_val + (rng.next() % range_size)
        numbers.add(num)
        attempts += 1
    
    return sorted(list(numbers))[:count]


if __name__ == '__main__':
    # Test all RNGs
    print("Testing all RNG types:\n")
    seed = 12345
    
    for rng_name in RNG_TYPES.keys():
        rng = create_rng(rng_name, seed)
        nums = generate_numbers(rng, 6, 1, 40)
        print(f"{rng_name:15s}: {nums}")
