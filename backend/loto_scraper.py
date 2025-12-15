#!/usr/bin/env python3
"""
Scraper pentru extrageri Loto 5/40 de pe noroc-chior.ro

Utilizare:
    python3 loto_scraper.py --year 2025
    python3 loto_scraper.py --year all
    python3 loto_scraper.py --year 2024,2023,2022
"""

import argparse
import json
import re
from datetime import datetime
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import time


class LotoScraper:
    def __init__(self):
        self.base_url = "http://noroc-chior.ro/Loto/5-din-40/arhiva-rezultate.php"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
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
            
            # Găsește tabelele - sunt 2: primul e header-ul cu rezultate recente, al doilea e arhiva
            tables = soup.find_all('table', class_='bilet')
            
            # Arhiva este al doilea tabel (dacă există)
            if len(tables) < 2:
                print(f"Nu s-a găsit tabel de arhivă pentru anul {year}")
                return []
            
            table = tables[1]  # Al doilea tabel este arhiva
            
            results = []
            rows = table.find_all('tr')
            
            # Sărește header-ul (primele 2 rânduri)
            for row in rows[2:]:
                cols = row.find_all('td')
                if len(cols) < 10:
                    continue
                
                # Extrage data
                date_text = cols[0].get_text(strip=True)
                
                # Extrage numerele (următoarele 6 coloane după dată)
                numbers = []
                for i in range(1, 7):
                    num_text = cols[i].get_text(strip=True)
                    if num_text.isdigit():
                        numbers.append(int(num_text))
                
                if len(numbers) == 6:
                    # Sortează numerele pentru consistență
                    numbers_sorted = sorted(numbers)
                    
                    # Parse data
                    date_obj = self._parse_date(date_text, year)
                    
                    results.append({
                        'date': date_obj,
                        'date_str': date_text,
                        'numbers': numbers,  # Ordinea extragerii
                        'numbers_sorted': numbers_sorted,  # Sortate
                        'year': year
                    })
            
            print(f"  ✓ Extrase {len(results)} extrageri pentru anul {year}")
            return results
            
        except Exception as e:
            print(f"  ✗ Eroare la extragerea anului {year}: {e}")
            return []
    
    def _parse_date(self, date_text: str, year: int) -> str:
        """
        Convertește textul datei în format ISO
        """
        # Mapare luni românești
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
        Salvează datele în format JSON
        """
        # Sortează după dată
        data_sorted = sorted(data, key=lambda x: x['date'])
        
        output = {
            'total_draws': len(data_sorted),
            'years': list(set([d['year'] for d in data_sorted])),
            'extracted_at': datetime.now().isoformat(),
            'draws': data_sorted
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Date salvate în: {filename}")
        print(f"  Total extrageri: {len(data_sorted)}")


def main():
    parser = argparse.ArgumentParser(
        description='Extrage rezultate Loto 5/40 de pe noroc-chior.ro'
    )
    parser.add_argument(
        '--year',
        type=str,
        default='2025',
        help='Anul sau anii de extras (ex: 2025, 2024,2023, sau "all" pentru 1995-2025)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/app/backend/loto_data.json',
        help='Fișier de ieșire JSON'
    )
    
    args = parser.parse_args()
    
    scraper = LotoScraper()
    
    # Determină anii de extras
    if args.year.lower() == 'all':
        years = list(range(1995, 2026))  # 1995-2025
        print("Extragere completă: 1995-2025")
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
        print("\n" + "="*50)
        print("STATISTICI RAPIDE")
        print("="*50)
        
        # Numără frecvența numerelor
        from collections import Counter
        all_numbers = []
        for draw in results:
            all_numbers.extend(draw['numbers'])
        
        counter = Counter(all_numbers)
        most_common = counter.most_common(10)
        
        print(f"\nTop 10 cele mai frecvente numere:")
        for num, count in most_common:
            print(f"  {num:2d}: apare de {count:3d} ori")
        
        print(f"\nTotal numere extrase: {len(all_numbers)}")
        print(f"Numere unice (1-40): {len(set(all_numbers))}")
    else:
        print("\n✗ Nu s-au putut extrage date.")


if __name__ == '__main__':
    main()
