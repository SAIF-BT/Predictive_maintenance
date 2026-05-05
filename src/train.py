import pandas as pd
import xgboost as xgb
import yaml  # You might need to pip install PyYAML
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import os
import json

def train_model():
    # --- NEW: Load Parameters from YAML ---
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    config = params['train']
    # --------------------------------------

    # Load feature-rich data
    data_path = os.path.join('data', 'processed', 'train_FD001_features.csv')
    df = pd.read_csv(data_path)

    # Group-based Split
    units = df['unit_nr'].unique()
    train_units = units[:80]
    val_units = units[80:]
    train_df = df[df['unit_nr'].isin(train_units)]
    val_df = df[df['unit_nr'].isin(val_units)]

    X_train = train_df.drop(columns=['unit_nr', 'time_cycles', 'RUL'])
    y_train = train_df['RUL']
    X_val = val_df.drop(columns=['unit_nr', 'time_cycles', 'RUL'])
    y_val = val_df['RUL']

    # --- UPDATED: Use config variables ---
    print(f"🚀 Training XGBoost (depth={config['max_depth']}, lr={config['learning_rate']})...")
    model = xgb.XGBRegressor(
        n_estimators=config['n_estimators'],
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        random_state=config['random_state']
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_val)
    rmse = float(root_mean_squared_error(y_val, preds))
    mae = float(mean_absolute_error(y_val, preds))

    # Save Artifacts
    os.makedirs('models', exist_ok=True)
    model.save_model('models/model_xgb.json')
    
    metrics = {"rmse": rmse, "mae": mae}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    print(f"✅ Training Complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}")

if __name__ == "__main__":
    train_model()