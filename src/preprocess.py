import pandas as pd
import numpy as np
import os
import yaml
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # Load config
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    dataset = params['data']['dataset']

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Check possible locations
    paths_to_check = [
        os.path.join(project_root, 'data', 'raw', f'train_{dataset}.txt'),
        os.path.join(project_root, 'data', 'raw', 'CMaps', f'train_{dataset}.txt')
    ]

    raw_file = next((p for p in paths_to_check if os.path.exists(p)), None)
    if not raw_file:
        print(f"❌ ERROR: train_{dataset}.txt not found.")
        return

    print(f"🔍 Processing {dataset} from: {raw_file}")
    
    cols = ['unit_nr', 'time_cycles', 'os_1', 'os_2', 'os_3'] + [f's_{i}' for i in range(1, 22)]
    df = pd.read_csv(raw_file, sep=r'\s+', header=None, names=cols, engine='python')
    
    # Target: Piecewise RUL
    df['RUL'] = df.groupby('unit_nr')['time_cycles'].transform(max) - df['time_cycles']
    df['RUL'] = df['RUL'].clip(upper=125)

    # FD002/004 Special: Regime Normalization
    # Clusters flight conditions so we scale sensors relative to their environment
    df['regime'] = df[['os_1', 'os_2', 'os_3']].round(1).astype(str).sum(axis=1)
    sensors = [f's_{i}' for i in range(1, 22)]
    scaler = StandardScaler()
    
    for regime in df['regime'].unique():
        mask = df['regime'] == regime
        # Some sensors are constant in certain regimes; we handle that automatically
        df.loc[mask, sensors] = scaler.fit_transform(df.loc[mask, sensors])

    output_path = os.path.join(project_root, 'data', 'processed', f'train_{dataset}_processed.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed {dataset} saved.")

if __name__ == "__main__":
    preprocess_data()