import pandas as pd
import os
import yaml

def create_features():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    dataset = params['data']['dataset']
    
    input_path = f'data/processed/train_{dataset}_processed.csv'
    df = pd.read_csv(input_path)
    
    sensor_cols = [f's_{i}' for i in range(1, 22)]
    
    # Calculate rolling metrics
    for col in sensor_cols:
        df[f'{col}_mean'] = df.groupby('unit_nr')[col].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        df[f'{col}_std'] = df.groupby('unit_nr')[col].transform(lambda x: x.rolling(window=10, min_periods=1).std().fillna(0))

    output_path = f'data/processed/train_{dataset}_features.csv'
    df.to_csv(output_path, index=False)
    print(f"✨ Features generated for {dataset}")

if __name__ == "__main__":
    create_features()