import pandas as pd
import xgboost as xgb
import yaml
import os

def predict_single_engine(unit_id=1):
    # 1. Load the Model
    model = xgb.XGBRegressor()
    model.load_model('models/model_xgb.json')
    
    # 2. Get the latest data for that engine
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    dataset = params['data']['dataset']
    
    df = pd.read_csv(f'data/processed/train_{dataset}_features.csv')
    engine_data = df[df['unit_nr'] == unit_id].iloc[-1:] # Take the most recent cycle
    
    # 3. Clean columns to match training
    drop_cols = ['unit_nr', 'time_cycles', 'RUL', 'regime']
    X = engine_data.drop(columns=[c for c in drop_cols if c in engine_data.columns])
    
    # 4. Predict
    prediction = model.predict(X)[0]
    
    print(f"--- 🛠️ Maintenance Alert (Dataset: {dataset}) ---")
    print(f"Engine ID: {unit_id}")
    print(f"Current Flight Cycle: {int(engine_data['time_cycles'].iloc[0])}")
    print(f"Estimated Remaining Useful Life: {prediction:.1f} cycles")
    print("---------------------------------------------")

if __name__ == "__main__":
    predict_single_engine()