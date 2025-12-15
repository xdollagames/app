#!/usr/bin/env python3
"""
Configurație unificată pentru toate loteriile
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class LotteryConfig:
    """Configurație pentru un tip de loterie"""
    name: str  # Ex: "Loto 5/40"
    short_name: str  # Ex: "5-40"
    url_path: str  # Ex: "5-din-40" (pentru URL)
    
    # Configurație extragere
    numbers_to_draw: int  # Câte numere se extrag
    min_number: int  # Cel mai mic număr posibil
    max_number: int  # Cel mai mare număr posibil
    
    # Pentru loterii cu format special (ex: Joker)
    is_composite: bool = False  # True dacă are mai multe componente
    composite_parts: List[Tuple[int, int, int]] = None  # [(count, min, max), ...]
    
    def __post_init__(self):
        if self.is_composite and self.composite_parts is None:
            raise ValueError(f"{self.name}: composite lottery must define composite_parts")
    
    @property
    def range_size(self) -> int:
        """Dimensiunea range-ului"""
        return self.max_number - self.min_number + 1
    
    @property
    def total_numbers(self) -> int:
        """Total numere extrase (inclusiv componente)"""
        if self.is_composite:
            return sum(part[0] for part in self.composite_parts)
        return self.numbers_to_draw


# Configurații pentru toate loteriile
LOTTERY_CONFIGS: Dict[str, LotteryConfig] = {
    '5-40': LotteryConfig(
        name='Loto 5/40',
        short_name='5-40',
        url_path='5-din-40',
        numbers_to_draw=6,  # De fapt extrage 6 (5 + 1 bonus)
        min_number=1,
        max_number=40
    ),
    
    '6-49': LotteryConfig(
        name='Loto 6/49',
        short_name='6-49',
        url_path='6-din-49',
        numbers_to_draw=6,
        min_number=1,
        max_number=49
    ),
    
    'joker': LotteryConfig(
        name='Joker',
        short_name='joker',
        url_path='joker',
        numbers_to_draw=6,  # 5 + 1 joker
        min_number=1,
        max_number=45,
        is_composite=True,
        composite_parts=[
            (5, 1, 45),  # 5 numere din 1-45
            (1, 1, 20),  # 1 număr "Joker" din 1-20
        ]
    ),
}


def get_lottery_config(lottery_type: str) -> LotteryConfig:
    """Obține configurația pentru un tip de loterie"""
    if lottery_type not in LOTTERY_CONFIGS:
        available = ', '.join(LOTTERY_CONFIGS.keys())
        raise ValueError(f"Unknown lottery type: {lottery_type}. Available: {available}")
    return LOTTERY_CONFIGS[lottery_type]


def list_available_lotteries() -> List[str]:
    """Listează toate loteriile disponibile"""
    return list(LOTTERY_CONFIGS.keys())


if __name__ == '__main__':
    print("Configurații Loterie Disponibile:\n")
    for key, config in LOTTERY_CONFIGS.items():
        print(f"{key:10s}: {config.name}")
        print(f"  Extrage: {config.numbers_to_draw} numere din {config.min_number}-{config.max_number}")
        if config.is_composite:
            print(f"  Format compus:")
            for i, (count, min_val, max_val) in enumerate(config.composite_parts, 1):
                print(f"    Partea {i}: {count} numere din {min_val}-{max_val}")
        print()
