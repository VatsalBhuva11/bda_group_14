import pandas as pd
import os

# List of CSV files to analyze
csv_files = ["CMC-2023-03-06-merged.csv", "data_who.csv", "FuelConsumptionCo2.csv"]

for csv_file in csv_files:
    if os.path.exists(csv_file):
        try:
            print(f"\n{'='*50}")
            print(f"File: {csv_file}")
            print(f"{'='*50}")
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Print shape
            print(f"Shape: {df.shape}")
            print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            # Print column names
            print(f"\nColumn Names ({len(df.columns)} columns):")
            for i, col in enumerate(df.columns, 1):
                print(f"{i:2d}. {col}")
            
            # Print data types
            print(f"\nData Types:")
            print(df.dtypes)
            
            # Print first few rows
            print(f"\nFirst 3 rows:")
            print(df.head(3))
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    else:
        print(f"File {csv_file} not found")

print(f"\n{'='*50}")
print("Analysis complete!")
