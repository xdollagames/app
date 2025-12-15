#!/usr/bin/env python3
"""
Multi-Game Configuration - Config pentru toate jocurile de noroc

Suportat:
- Loto 5/40
- Loto 6/49
- Joker
- Noroc
- Noroc Plus
"""

GAME_CONFIGS = {
    'loto_5_40': {
        'name': 'Loto 5/40',
        'numbers': 6,
        'min_number': 1,
        'max_number': 40,
        'scraper_url': 'http://noroc-chior.ro/Loto/5-din-40/arhiva-rezultate.php',
        'table_class': 'bilet',
        'number_columns': [1, 2, 3, 4, 5, 6],  # Coloanele cu numere în tabel
    },
    'loto_6_49': {
        'name': 'Loto 6/49',
        'numbers': 6,
        'min_number': 1,
        'max_number': 49,
        'scraper_url': 'http://noroc-chior.ro/Loto/6-din-49/arhiva-rezultate.php',
        'table_class': 'bilet',
        'number_columns': [1, 2, 3, 4, 5, 6],
    },
    'joker': {
        'name': 'Joker',
        'numbers': 5,  # 5 cifre principale
        'min_number': 0,
        'max_number': 9,
        'has_bonus': True,  # Are cifră bonus
        'scraper_url': 'http://noroc-chior.ro/Joker/arhiva-rezultate.php',
        'table_class': 'bilet',
        'number_columns': [1, 2, 3, 4, 5],  # Primele 5 cifre
        'bonus_column': 6,  # Cifra Joker
    },
    'noroc': {
        'name': 'Noroc',
        'numbers': 6,  # 6 cifre
        'min_number': 0,
        'max_number': 9,
        'scraper_url': 'http://noroc-chior.ro/Noroc/arhiva-rezultate.php',
        'table_class': 'bilet',
        'number_columns': [1, 2, 3, 4, 5, 6],
    },
    'noroc_plus': {
        'name': 'Noroc Plus',
        'numbers': 6,
        'min_number': 0,
        'max_number': 9,
        'scraper_url': 'http://noroc-chior.ro/Noroc-Plus/arhiva-rezultate.php',
        'table_class': 'bilet',
        'number_columns': [1, 2, 3, 4, 5, 6],
    },
}


def get_game_config(game_type: str) -> dict:
    """Returnează configurația pentru un joc"""
    if game_type not in GAME_CONFIGS:
        raise ValueError(f"Unknown game type: {game_type}. Available: {list(GAME_CONFIGS.keys())}")
    return GAME_CONFIGS[game_type]


def list_games():
    """Listează toate jocurile disponibile"""
    print("\nJocuri disponibile:\n")
    for key, config in GAME_CONFIGS.items():
        print(f"  {key:15s}: {config['name']}")
        print(f"                   {config['numbers']} numere din {config['min_number']}-{config['max_number']}")
        if config.get('has_bonus'):
            print(f"                   + cifră bonus")
        print()


if __name__ == '__main__':
    list_games()
