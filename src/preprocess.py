import pandas as pd
import numpy as np
import os
import sys

def preprocess_nasa_data(file_number="FD001"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Check two possible locations for the file
    paths_to_check = [
        os.path.join(project_root, 'data', 'raw', f'train_{file_number}.txt'),
        os.path.join(project_root, 'data', 'raw', 'CMaps', f'train_{file_number}.txt')
    ]

    raw_file = None
    for path in paths_to_check:
        if os.path.exists(path):
            raw_file = path
            break

    if not raw_file:
        print(f"❌ ERROR: train_{file_number}.txt not found in data/raw or data/raw/CMaps")
        print(f"Current Directory: {os.getcwd()}")
        if os.path.exists(os.path.join(project_root, 'data', 'raw')):
            print(f"Contents of data/raw: {os.listdir(os.path.join(project_root, 'data', 'raw'))}")
        sys.exit(1)

    print(f"🔍 Found file at: {raw_file}")
    
    # Load and process
    all_cols = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    df = pd.read_csv(raw_file, sep=r'\s+', header=None, names=all_cols, engine='python')
    
    # Calculate Piecewise Linear RUL
    max_cycle = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_life']
    df = df.merge(max_cycle, on='unit_nr', how='left')
    df['RUL'] = (df['max_life'] - df['time_cycles']).clip(upper=125)
    df.drop(columns=['max_life'], inplace=True)

    output_file = os.path.join(project_root, 'data', 'processed', f'train_{file_number}_processed.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Success! Saved to {output_file}")

if __name__ == "__main__":
    preprocess_nasa_data()