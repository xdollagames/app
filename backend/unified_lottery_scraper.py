#!/usr/bin/env python3
"""
Scraper unificat pentru toate loteriile suportate

Utilizare:
    python3 unified_lottery_scraper.py --lottery 6-49 --year 2025
    python3 unified_lottery_scraper.py --lottery joker --year all
    python3 unified_lottery_scraper.py --lottery 5-40 --year 2024,2023
"""

import argparse
import json
import re
from datetime import datetime
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import time

from lottery_config import get_lottery_config, list_available_lotteries


class UnifiedLotteryScraper:
    def __init__(self, lottery_type: str):
        self.config = get_lottery_config(lottery_type)
        self.base_url = f"http://noroc-chior.ro/Loto/{self.config.url_path}/arhiva-rezultate.php"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        print(f"Scraper inițializat pentru: {self.config.name}")
        print(f"URL: {self.base_url}\n")
    
    def scrape_year(self, year: int) -> List[Dict]:
        """
        Extrage toate extragerile pentru un an dat
        """
        url = f"{self.base_url}?Y={year}"
        print(f"Extragere date pentru anul {year}...")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Găsește tabelele - al doilea tabel e arhiva
            tables = soup.find_all('table', class_='bilet')
            
            if len(tables) < 2:
                print(f"Nu s-a găsit tabel de arhivă pentru anul {year}")
                return []
            
            table = tables[1]  # Al doilea tabel
            
            results = []
            rows = table.find_all('tr')
            
            # Sărește header-ul (primele 2 rânduri)
            for row in rows[2:]:
                cols = row.find_all('td')
                if len(cols) < 10:
                    continue
                
                # Extrage data
                date_text = cols[0].get_text(strip=True)
                
                # Extrage numerele
                numbers = self._extract_numbers(cols)
                
                if numbers and len(numbers) == self.config.numbers_to_draw:
                    # Parse data
                    date_obj = self._parse_date(date_text, year)
                    
                    result = {
                        'date': date_obj,
                        'date_str': date_text,
                        'numbers': numbers,
                        'numbers_sorted': sorted(numbers),
                        'year': year,
                        'lottery_type': self.config.short_name
                    }
                    
                    # Pentru loterii compuse, adaugă breakdown
                    if self.config.is_composite:
                        result['composite_breakdown'] = self._breakdown_composite(numbers)
                    
                    results.append(result)
            
            print(f"  ✓ Extrase {len(results)} extrageri pentru anul {year}")
            return results
            
        except Exception as e:
            print(f"  ✗ Eroare la extragerea anului {year}: {e}")
            return []
    
    def _extract_numbers(self, cols) -> List[int]:
        """
        Extrage numerele din coloanele HTML
        """
        numbers = []
        
        # Majoritatea loteriilor: numerele sunt în coloanele 1-6 (sau 1-7)
        # Începem de la coloana 1 (după dată)
        for i in range(1, min(10, len(cols))):
            num_text = cols[i].get_text(strip=True)
            if num_text.isdigit():
                num = int(num_text)
                # Validare range
                if self.config.min_number <= num <= self.config.max_number:
                    numbers.append(num)
                # Pentru Joker, al 6-lea număr e în range diferit
                elif self.config.is_composite and len(numbers) == 5:
                    # Verifică dacă e în range-ul părții a 2-a
                    joker_min = self.config.composite_parts[1][1]
                    joker_max = self.config.composite_parts[1][2]
                    if joker_min <= num <= joker_max:
                        numbers.append(num)
            
            if len(numbers) == self.config.numbers_to_draw:
                break
        
        return numbers
    
    def _breakdown_composite(self, numbers: List[int]) -> Dict:
        """
        Pentru loterii compuse, împarte numerele în componente
        """
        breakdown = {}
        idx = 0
        
        for i, (count, min_val, max_val) in enumerate(self.config.composite_parts, 1):
            part_numbers = numbers[idx:idx + count]
            breakdown[f'part_{i}'] = {
                'numbers': part_numbers,
                'range': f'{min_val}-{max_val}',
                'description': f'{count} din {min_val}-{max_val}'
            }
            idx += count
        
        return breakdown
    
    def _parse_date(self, date_text: str, year: int) -> str:
        """
        Convertește textul datei în format ISO
        """
        months = {
            'ianuarie': 1, 'februarie': 2, 'martie': 3, 'aprilie': 4,
            'mai': 5, 'iunie': 6, 'iulie': 7, 'august': 8,
            'septembrie': 9, 'octombrie': 10, 'noiembrie': 11, 'decembrie': 12
        }
        
        # Pattern: "Du, 14 decembrie 2025"
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
        """
        Extrage date pentru mai mulți ani
        """
        all_results = []
        
        for year in years:
            results = self.scrape_year(year)
            all_results.extend(results)
            time.sleep(1)  # Respectăm serverul
        
        return all_results
    
    def save_to_json(self, data: List[Dict], filename: str):
        """
        Salvează datele în format JSON cu MERGE INCREMENTAL
        """
        data_sorted = sorted(data, key=lambda x: x['date'])
        
        # SALVARE cu MERGE INCREMENTAL
        # Dacă fișierul există, păstrăm extragerile vechi și adăugăm doar pe cele noi
        existing_draws = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_draws = existing_data.get('draws', [])
                print(f"\n📦 Fișier existent găsit: {len(existing_draws)} extrageri vechi")
        except:
            print(f"\n📦 Fișier nou - nu există date anterioare")
        
        # Merge: adaugă doar extragerile NOI (pe bază de dată)
        existing_dates = {d.get('date', d.get('date_str')): d for d in existing_draws}
        new_count = 0
        
        for draw in data_sorted:
            draw_date = draw.get('date', draw.get('date_str'))
            if draw_date not in existing_dates:
                # Nou!
                existing_draws.append(draw)
                new_count += 1
            else:
                # Actualizează (poate au fost corecții)
                existing_dates[draw_date] = draw
        
        # Sortează după dată
        existing_draws.sort(key=lambda x: x.get('date', x.get('date_str', '')))
        
        output = {
            'lottery_type': self.config.short_name,
            'lottery_name': self.config.name,
            'config': {
                'numbers_to_draw': self.config.numbers_to_draw,
                'min_number': self.config.min_number,
                'max_number': self.config.max_number,
                'is_composite': self.config.is_composite
            },
            'total_draws': len(existing_draws),
            'years': sorted(list(set([d['year'] for d in existing_draws]))),
            'extracted_at': datetime.now().isoformat(),
            'draws': existing_draws
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Date salvate în: {filename}")
        print(f"  Total extrageri: {len(existing_draws)}")
        if new_count > 0:
            print(f"  ✅ Extrageri NOI adăugate: {new_count}")
        else:
            print(f"  ℹ️  Nicio extragere nouă (totul up-to-date)")


def main():
    parser = argparse.ArgumentParser(
        description='Scraper unificat pentru toate loteriile'
    )
    parser.add_argument(
        '--lottery',
        type=str,
        required=True,
        choices=list_available_lotteries(),
        help='Tipul de loterie'
    )
    parser.add_argument(
        '--year',
        type=str,
        default='2025',
        help='Anul sau anii de extras (ex: 2025, 2024,2023, sau "all" pentru toate)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Fișier de ieșire JSON (default: {lottery}_data.json)'
    )
    
    args = parser.parse_args()
    
    # Default output filename
    if args.output is None:
        args.output = f'/app/backend/{args.lottery}_data.json'
    
    scraper = UnifiedLotteryScraper(args.lottery)
    
    # Determină anii de extras
    if args.year.lower() == 'all':
        years = list(range(1995, 2026))
        print(f"Extragere completă: 1995-2025")
        print("AVERTISMENT: Acest proces poate dura câteva minute...\n")
    elif ',' in args.year:
        years = [int(y.strip()) for y in args.year.split(',')]
    else:
        years = [int(args.year)]
    
    # Extrage datele
    results = scraper.scrape_multiple_years(years)
    
    if results:
        # Salvează în JSON
        scraper.save_to_json(results, args.output)
        
        # Afișează statistici rapide
        print(f"\n{'='*50}")
        print("STATISTICI RAPIDE")
        print(f"{'='*50}")
        
        from collections import Counter
        
        # Pentru loterii compuse, analizează separat
        if scraper.config.is_composite:
            print(f"\n{scraper.config.name} - Analiză pe componente:\n")
            for i, (count, min_val, max_val) in enumerate(scraper.config.composite_parts, 1):
                print(f"Partea {i}: {count} numere din {min_val}-{max_val}")
                part_numbers = []
                for draw in results:
                    if 'composite_breakdown' in draw:
                        part_numbers.extend(draw['composite_breakdown'][f'part_{i}']['numbers'])
                
                counter = Counter(part_numbers)
                most_common = counter.most_common(10)
                print(f"  Top 10:")
                for num, freq in most_common:
                    print(f"    {num:2d}: apare de {freq:3d} ori")
                print()
        else:
            all_numbers = []
            for draw in results:
                all_numbers.extend(draw['numbers'])
            
            counter = Counter(all_numbers)
            most_common = counter.most_common(10)
            
            print(f"\nTop 10 cele mai frecvente numere:")
            for num, count in most_common:
                print(f"  {num:2d}: apare de {count:3d} ori")
            
            print(f"\nTotal numere extrase: {len(all_numbers)}")
            print(f"Numere unice ({scraper.config.min_number}-{scraper.config.max_number}): {len(set(all_numbers))}")
    else:
        print("\n✗ Nu s-au putut extrage date.")


if __name__ == '__main__':
    main()
