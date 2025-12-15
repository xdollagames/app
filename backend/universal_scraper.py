#!/usr/bin/env python3
"""
Universal Scraper - Pentru TOATE jocurile de noroc

Utilizare:
    # Loto 5/40
    python3 universal_scraper.py --game loto_5_40 --year 2024
    
    # Loto 6/49
    python3 universal_scraper.py --game loto_6_49 --year 2024
    
    # Joker
    python3 universal_scraper.py --game joker --year 2024
    
    # Toate jocurile
    python3 universal_scraper.py --game all --year 2024
"""

import argparse
import json
import re
import time
from datetime import datetime
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

from multi_game_config import get_game_config, GAME_CONFIGS, list_games


class UniversalScraper:
    def __init__(self, game_type: str):
        self.game_type = game_type
        self.config = get_game_config(game_type)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        print(f"\n🎲 Game: {self.config['name']}")
        print(f"   {self.config['numbers']} numere din {self.config['min_number']}-{self.config['max_number']}")
        if self.config.get('has_bonus'):
            print(f"   + cifră bonus")
        print()
    
    def scrape_year(self, year: int) -> List[Dict]:
        """Extrage toate extragerile pentru un an"""
        url = f"{self.config['scraper_url']}?Y={year}"
        print(f"Extragere date pentru anul {year}...")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Găsește tabelele
            tables = soup.find_all('table', class_=self.config['table_class'])
            
            if len(tables) < 2:
                print(f"  Nu s-a găsit tabel de arhivă pentru anul {year}")
                return []
            
            table = tables[1]  # Al doilea tabel e arhiva
            results = []
            rows = table.find_all('tr')
            
            # Sărește header-ul
            for row in rows[2:]:
                cols = row.find_all('td')
                if len(cols) < max(self.config['number_columns']) + 1:
                    continue
                
                # Extrage data
                date_text = cols[0].get_text(strip=True)
                
                # Extrage numerele
                numbers = []
                for col_idx in self.config['number_columns']:
                    num_text = cols[col_idx].get_text(strip=True)
                    if num_text.isdigit():
                        numbers.append(int(num_text))
                
                # Extrage bonus dacă există
                bonus = None
                if self.config.get('has_bonus'):
                    bonus_col = self.config.get('bonus_column')
                    if bonus_col and len(cols) > bonus_col:
                        bonus_text = cols[bonus_col].get_text(strip=True)
                        if bonus_text.isdigit():
                            bonus = int(bonus_text)
                
                if len(numbers) == self.config['numbers']:
                    # Pentru jocuri cu sortare (Loto)
                    if self.config['max_number'] > 10:  # Loto type
                        numbers_sorted = sorted(numbers)
                    else:  # Cifre (Joker, Noroc) - păstrează ordinea!
                        numbers_sorted = numbers
                    
                    result = {
                        'date': self._parse_date(date_text, year),
                        'date_str': date_text,
                        'numbers': numbers,
                        'numbers_sorted': numbers_sorted,
                        'year': year
                    }
                    
                    if bonus is not None:
                        result['bonus'] = bonus
                    
                    results.append(result)
            
            print(f"  ✓ Extrase {len(results)} extrageri pentru anul {year}")
            return results
            
        except Exception as e:
            print(f"  ✗ Eroare la extragerea anului {year}: {e}")
            return []
    
    def _parse_date(self, date_text: str, year: int) -> str:
        """Convertește textul datei în format ISO"""
        months = {
            'ianuarie': 1, 'februarie': 2, 'martie': 3, 'aprilie': 4,
            'mai': 5, 'iunie': 6, 'iulie': 7, 'august': 8,
            'septembrie': 9, 'octombrie': 10, 'noiembrie': 11, 'decembrie': 12
        }
        
        pattern = r'\w+,\s+(\d+)\s+(\w+)\s+(\d+)'
        match = re.search(pattern, date_text)
        
        if match:
            day = int(match.group(1))
            month_name = match.group(2).lower()
            year_extracted = int(match.group(3))
            month = months.get(month_name, 1)
            
            try:
                date_obj = datetime(year_extracted, month, day)
                return date_obj.strftime('%Y-%m-%d')
            except:
                return f"{year}-01-01"
        
        return f"{year}-01-01"
    
    def scrape_multiple_years(self, years: List[int]) -> List[Dict]:
        """Extrage date pentru mai mulți ani"""
        all_results = []
        
        for year in years:
            results = self.scrape_year(year)
            all_results.extend(results)
            time.sleep(1)
        
        return all_results
    
    def save_to_json(self, data: List[Dict], filename: str):
        """Salvează datele în format JSON"""
        data_sorted = sorted(data, key=lambda x: x['date'])
        
        output = {
            'game': self.game_type,
            'game_name': self.config['name'],
            'total_draws': len(data_sorted),
            'years': list(set([d['year'] for d in data_sorted])),
            'extracted_at': datetime.now().isoformat(),
            'config': {
                'numbers': self.config['numbers'],
                'min_number': self.config['min_number'],
                'max_number': self.config['max_number'],
                'has_bonus': self.config.get('has_bonus', False)
            },
            'draws': data_sorted
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Date salvate în: {filename}")
        print(f"  Total extrageri: {len(data_sorted)}")


def main():
    parser = argparse.ArgumentParser(
        description='Universal Scraper - Pentru TOATE jocurile de noroc',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--game',
        type=str,
        required=True,
        help='Tipul jocului: loto_5_40, loto_6_49, joker, noroc, noroc_plus, sau all'
    )
    parser.add_argument(
        '--year',
        type=str,
        default='2024',
        help='Anul sau anii: 2024, 2024,2023, sau "all" pentru 1995-2025'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Fișier output (default: {game}_data.json)'
    )
    parser.add_argument(
        '--list-games',
        action='store_true',
        help='Listează toate jocurile disponibile'
    )
    
    args = parser.parse_args()
    
    if args.list_games:
        list_games()
        return
    
    print("\n" + "="*70)
    print("  UNIVERSAL SCRAPER - Multi-Game Support")
    print("="*70)
    
    # Determină anii
    if args.year.lower() == 'all':
        years = list(range(1995, 2026))
        print(f"\nExtragere completă: 1995-2025 ({len(years)} ani)")
    elif ',' in args.year:
        years = [int(y.strip()) for y in args.year.split(',')]
    else:
        years = [int(args.year)]
    
    # Determină jocurile
    if args.game.lower() == 'all':
        games = list(GAME_CONFIGS.keys())
        print(f"\nExtragere pentru TOATE jocurile ({len(games)})\n")
    else:
        games = [args.game]
    
    # Extrage date pentru fiecare joc
    for game in games:
        try:
            scraper = UniversalScraper(game)
            results = scraper.scrape_multiple_years(years)
            
            if results:
                output_file = args.output or f"{game}_data.json"
                scraper.save_to_json(results, output_file)
                
                # Statistici rapide
                print("\n" + "="*50)
                print("STATISTICI RAPIDE")
                print("="*50)
                
                from collections import Counter
                all_numbers = []
                for draw in results:
                    all_numbers.extend(draw['numbers'])
                
                counter = Counter(all_numbers)
                most_common = counter.most_common(10)
                
                print(f"\nTop 10 cele mai frecvente numere/cifre:")
                for num, count in most_common:
                    print(f"  {num:2d}: apare de {count:3d} ori")
            else:
                print(f"\n✗ Nu s-au putut extrage date pentru {game}")
        
        except Exception as e:
            print(f"\n✗ Eroare la procesarea {game}: {e}")
            continue


if __name__ == '__main__':
    main()
