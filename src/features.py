import pandas as pd
import numpy as np
import os

def generate_features(file_number="FD001"):
    # Ensure we use absolute paths to avoid PowerShell/DVC confusion
    project_root = os.getcwd()
    input_file = os.path.join(project_root, 'data', 'processed', f'train_{file_number}_processed.csv')
    output_file = os.path.join(project_root, 'data', 'processed', f'train_{file_number}_features.csv')
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    sensor_cols = [col for col in df.columns if col.startswith('s_')]
    
    window = 10
    for col in sensor_cols:
        df[f'{col}_mean'] = df.groupby('unit_nr')[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'{col}_std'] = df.groupby('unit_nr')[col].transform(lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))

    df.to_csv(output_file, index=False)
    print(f"✅ Features generated: {output_file}")

if __name__ == "__main__":
    generate_features()