#!/usr/bin/env python3
"""
Convert DMSC TripAdvisor data to CSV format.

Original format:
    ratings (7 aspects) <TAB><TAB> review text with <ssssss> separators

Output format:
    CSV with columns: review_text, rating, sentiment, aspect_ratings
"""

import os
import csv
from typing import List, Tuple


ASPECT_NAMES = [
    'value',      # value, price, quality, worth
    'room',       # room, suite, view, bed
    'location',   # location, traffic, restaurant
    'cleanliness',# clean, dirty, smell
    'staff',      # staff, check-in, help
    'service',    # service, food, breakfast
    'business'    # business, internet
]


def parse_line(line: str) -> dict:
    """Parse a single line from DMSC format."""
    line = line.strip()
    if not line:
        return None

    # Split by double tab
    parts = line.split('\t\t')
    if len(parts) != 2:
        # Try single tab
        parts = line.split('\t')
        if len(parts) < 2:
            return None

    # Parse ratings
    rating_str = parts[0].strip()
    text = parts[1].strip() if len(parts) > 1 else ""

    ratings = []
    for r in rating_str.split():
        try:
            ratings.append(int(r))
        except:
            ratings.append(-1)

    # Pad or truncate to 7 ratings
    while len(ratings) < 7:
        ratings.append(-1)
    ratings = ratings[:7]

    # Calculate average rating (excluding -1)
    valid_ratings = [r for r in ratings if r != -1]
    avg_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else 3.0

    # Binary sentiment: >= 4 is positive
    sentiment = 1 if avg_rating >= 4 else 0

    # Clean text: replace <ssssss> with space
    clean_text = text.replace('<ssssss>', ' ').strip()
    # Remove multiple spaces
    clean_text = ' '.join(clean_text.split())

    return {
        'review_text': clean_text,
        'avg_rating': round(avg_rating, 2),
        'sentiment': sentiment,
        'value': ratings[0],
        'room': ratings[1],
        'location': ratings[2],
        'cleanliness': ratings[3],
        'staff': ratings[4],
        'service': ratings[5],
        'business': ratings[6],
    }


def convert_file(input_path: str, output_path: str) -> int:
    """Convert a single file to CSV."""
    records = []

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            record = parse_line(line)
            if record:
                records.append(record)

    if not records:
        print(f"  No valid records found in {input_path}")
        return 0

    # Write CSV
    fieldnames = ['review_text', 'avg_rating', 'sentiment',
                  'value', 'room', 'location', 'cleanliness',
                  'staff', 'service', 'business']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return len(records)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    files_to_convert = ['train', 'dev', 'test']

    print("Converting DMSC TripAdvisor data to CSV format...")
    print("=" * 60)

    total_records = 0
    all_records = []

    for filename in files_to_convert:
        input_path = os.path.join(script_dir, filename)
        output_path = os.path.join(script_dir, f'{filename}.csv')

        if os.path.exists(input_path):
            count = convert_file(input_path, output_path)
            print(f"  {filename}: {count:,} records -> {filename}.csv")
            total_records += count

            # Also collect for combined file
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    record = parse_line(line)
                    if record:
                        record['split'] = filename
                        all_records.append(record)
        else:
            print(f"  {filename}: NOT FOUND")

    # Create combined file
    combined_path = os.path.join(script_dir, 'combined.csv')
    fieldnames = ['review_text', 'avg_rating', 'sentiment',
                  'value', 'room', 'location', 'cleanliness',
                  'staff', 'service', 'business', 'split']

    with open(combined_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("=" * 60)
    print(f"Total: {total_records:,} records")
    print(f"Combined file: combined.csv ({len(all_records):,} records)")

    # Print class distribution
    positive = sum(1 for r in all_records if r['sentiment'] == 1)
    negative = sum(1 for r in all_records if r['sentiment'] == 0)
    print(f"\nClass Distribution:")
    print(f"  Positive: {positive:,} ({positive/len(all_records)*100:.1f}%)")
    print(f"  Negative: {negative:,} ({negative/len(all_records)*100:.1f}%)")


if __name__ == "__main__":
    main()
