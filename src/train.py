import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import json

def train_model():
    # 1. Load parameters from YAML
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    dataset = params['data']['dataset']  # This will now be 'FD002'
    config = params['train']

    # 2. Load the CORRECT feature file
    data_path = os.path.join('data', 'processed', f'train_{dataset}_features.csv')
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: {data_path} not found!")
        return

    df = pd.read_csv(data_path)
    
    # 3. Group-based Split (Keep units together)
    units = df['unit_nr'].unique()
    train_units = units[:int(len(units)*0.8)] # Use 80% for training
    val_units = units[int(len(units)*0.8):]
    
    train_df = df[df['unit_nr'].isin(train_units)]
    val_df = df[df['unit_nr'].isin(val_units)]

    # Drop non-feature columns
    # Note: we also drop 'regime' if it exists in the features file
    drop_cols = ['unit_nr', 'time_cycles', 'RUL', 'regime']
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df['RUL']
    
    X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    y_val = val_df['RUL']

    # 4. Train
    print(f"🚀 Training XGBoost on {dataset}...")
    model = xgb.XGBRegressor(
        n_estimators=config['n_estimators'],
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        random_state=config['random_state']
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    preds = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))

    # 6. Save Artifacts
    os.makedirs('models', exist_ok=True)
    model.save_model('models/model_xgb.json')
    
    metrics = {"rmse": rmse, "mae": mae}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    print(f"✅ Training Complete for {dataset}. RMSE: {rmse:.2f}, MAE: {mae:.2f}")

if __name__ == "__main__":
    train_model()
