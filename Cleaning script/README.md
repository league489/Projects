# California Housing - Data Cleaning

This script cleans the California Housing dataset.

## What it does
- Removes duplicates
- Fills missing values in numeric columns with mean
- Drops missing values in categorical columns
- Normalizes numerical columns (Min-Max)
- Saves cleaned data to a new file

## How to run
python clean.py input.csv output.csv

## Results
After cleaning, all numerical columns are in range [0,1].
