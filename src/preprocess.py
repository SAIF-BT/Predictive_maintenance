import pandas as pd
import numpy as np
import os
import yaml
import pickle
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    dataset = params['data']['dataset']

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    paths_to_check = [
        os.path.join(project_root, 'data', 'raw', f'train_{dataset}.txt'),
        os.path.join(project_root, 'data', 'raw', 'CMaps', f'train_{dataset}.txt')
    ]

    raw_file = next((p for p in paths_to_check if os.path.exists(p)), None)
    if not raw_file:
        raise FileNotFoundError(
            f"train_{dataset}.txt not found in {paths_to_check}"
        )

    print(f"Processing {dataset} from: {raw_file}")

    cols = ['unit_nr', 'time_cycles', 'os_1', 'os_2', 'os_3'] + [f's_{i}' for i in range(1, 22)]
    df = pd.read_csv(raw_file, sep=r'\s+', header=None, names=cols, engine='python')

    df['RUL'] = df.groupby('unit_nr')['time_cycles'].transform(max) - df['time_cycles']
    df['RUL'] = df['RUL'].clip(upper=125)

    df['regime'] = df[['os_1', 'os_2', 'os_3']].round(1).astype(str).sum(axis=1)
    sensors = [f's_{i}' for i in range(1, 22)]

    # ✅ separate scaler per regime
    scalers = {}
    for regime in df['regime'].unique():
        mask = df['regime'] == regime
        s = StandardScaler()
        df.loc[mask, sensors] = s.fit_transform(df.loc[mask, sensors])
        scalers[regime] = s

    # ✅ save scalers.pkl
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    scalers_path = os.path.join(models_dir, 'scalers.pkl')
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to: {scalers_path}")

    output_path = os.path.join(project_root, 'data', 'processed', f'train_{dataset}_processed.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed {dataset} saved.")

if __name__ == "__main__":
    preprocess_data()
