#!/usr/bin/env python3
"""
Verification Tool - Verifică predicțiile cu extragerile reale

Compară predicțiile generate cu rezultatele reale și calculează accuracy.

Utilizare:
    python3 verify_predictions.py --prediction pragmatic_results.json --actual "3,12,19,24,31,38"
    python3 verify_predictions.py --evaluate-history pragmatic_results.json
"""

import argparse
import json
from typing import List
from collections import Counter


def parse_numbers(numbers_str: str) -> List[int]:
    """Parse string to list of numbers"""
    return sorted([int(x.strip()) for x in numbers_str.split(',')])


def calculate_accuracy(predicted: List[int], actual: List[int]) -> dict:
    """Calculate various accuracy metrics"""
    pred_set = set(predicted)
    actual_set = set(actual)
    
    matches = len(pred_set & actual_set)
    score = matches / len(actual_set)
    
    return {
        'matches': matches,
        'score': score,
        'percentage': score * 100,
        'predicted': predicted,
        'actual': actual,
        'correct_numbers': sorted(list(pred_set & actual_set)),
        'missed_numbers': sorted(list(actual_set - pred_set)),
        'wrong_predictions': sorted(list(pred_set - actual_set))
    }


def evaluate_historical_performance(results_file: str):
    """
    Evaluează cum ar fi funcționat predicțiile pe date istorice
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    print(f"\n{'='*70}")
    print("HISTORICAL PERFORMANCE EVALUATION")
    print(f"{'='*70}\n")
    
    for rng_name, rng_data in results.items():
        if 'seed_sequence' not in rng_data:
            continue
        
        print(f"\nRNG: {rng_name}")
        print(f"Success Rate: {rng_data['success_rate']:.1%}")
        print(f"-" * 70)
        
        seed_seq = rng_data['seed_sequence']
        
        # Analizează match distribution
        match_dist = Counter([s['matches'] for s in seed_seq])
        
        print(f"\nMatch Distribution:")
        for matches in sorted(match_dist.keys(), reverse=True):
            count = match_dist[matches]
            pct = count / len(seed_seq) * 100
            bar = '█' * int(pct / 2)
            print(f"  {matches}/6: {bar} {count:3d} ({pct:5.1f}%)")
        
        # Best consecutive streak
        best_streak = 0
        current_streak = 0
        
        for s in seed_seq:
            if s['matches'] >= 4:
                current_streak += 1
                best_streak = max(best_streak, current_streak)
            else:
                current_streak = 0
        
        print(f"\nBest streak (4+/6 matches): {best_streak} consecutive")
        
        # Sample några predictions
        if len(seed_seq) >= 10:
            print(f"\nSample predictions (last 10):")
            for i, s in enumerate(seed_seq[-10:], 1):
                print(f"  {i:2d}. {s['date'][:15]:15s}: {s['matches']}/6 | Seed: {s['seed']:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Verify predictions against actual results'
    )
    parser.add_argument('--prediction', type=str,
                       help='Prediction file (pragmatic_results.json)')
    parser.add_argument('--actual', type=str,
                       help='Actual numbers (comma-separated)')
    parser.add_argument('--evaluate-history', type=str,
                       help='Evaluate historical performance')
    
    args = parser.parse_args()
    
    if args.evaluate_history:
        evaluate_historical_performance(args.evaluate_history)
    
    elif args.prediction and args.actual:
        with open(args.prediction, 'r') as f:
            data = json.load(f)
        
        actual_numbers = parse_numbers(args.actual)
        
        print(f"\n{'='*70}")
        print("PREDICTION VERIFICATION")
        print(f"{'='*70}")
        print(f"\nActual draw: {actual_numbers}")
        
        if 'predictions' in data and data['predictions']:
            predictions = data['predictions']
            
            print(f"\nEvaluating {len(predictions)} prediction(s):\n")
            
            best_accuracy = 0
            best_pred = None
            
            for i, pred in enumerate(predictions, 1):
                accuracy = calculate_accuracy(pred['numbers'], actual_numbers)
                
                print(f"{i}. {pred['method'].upper()}")
                print(f"   Predicted: {pred['numbers']}")
                print(f"   Matches: {accuracy['matches']}/6 ({accuracy['percentage']:.1f}%)")
                
                if accuracy['correct_numbers']:
                    print(f"   ✓ Correct: {accuracy['correct_numbers']}")
                if accuracy['missed_numbers']:
                    print(f"   ✗ Missed: {accuracy['missed_numbers']}")
                
                print(f"   Confidence: {pred['confidence']:.1%}")
                print()
                
                if accuracy['score'] > best_accuracy:
                    best_accuracy = accuracy['score']
                    best_pred = i
            
            print(f"{'='*70}")
            print(f"RESULT: Best prediction #{best_pred} with {best_accuracy*100:.1f}% accuracy")
            
            if best_accuracy >= 0.5:
                print(f"✓ GOOD RESULT! 3+/6 matches")
            elif best_accuracy >= 0.33:
                print(f"~ DECENT! 2/6 matches")
            else:
                print(f"✗ POOR! 0-1/6 matches")
            
            print(f"\nInterpretation:")
            if best_accuracy >= 0.5:
                print(f"  → Pattern working! Continue testing!")
            else:
                print(f"  → Pattern not working for this draw")
                print(f"  → Need more data or different approach")
        
        else:
            print("\nNo predictions found in file!")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
