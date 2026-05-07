import pandas as pd
import xgboost as xgb
import yaml
import os

def predict_single_engine(unit_id=1):

    # 1. Load params first                         # ✅ BUG 4 fixed — logical order
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    dataset = params['data']['dataset']

    # 2. Validate files exist before loading       # ✅ BUG 1 & 3 fixed
    model_path = 'models/model_xgb.json'
    data_path = f'data/processed/train_{dataset}_features.csv'

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run the train stage first."
        )
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Run the features stage first."
        )

    # 3. Load model and data
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    df = pd.read_csv(data_path)

    # 4. Validate unit_id exists                   # ✅ BUG 2 fixed
    available = df['unit_nr'].unique()
    if unit_id not in available:
        raise ValueError(
            f"Engine {unit_id} not found. "
            f"Available engines: {sorted(available)}"
        )

    # 5. Get most recent cycle for this engine
    engine_data = df[df['unit_nr'] == unit_id].iloc[-1:]

    # 6. Prepare features
    drop_cols = ['unit_nr', 'time_cycles', 'RUL', 'regime']
    X = engine_data.drop(columns=[c for c in drop_cols if c in engine_data.columns])

    # 7. Predict
    prediction = model.predict(X)[0]

    print("---------------------------------------------")
    print(f"Maintenance Alert (Dataset: {dataset})")
    print(f"Engine ID      : {unit_id}")
    print(f"Flight Cycle   : {int(engine_data['time_cycles'].iloc[0])}")
    print(f"Predicted RUL  : {prediction:.1f} cycles")
    print("---------------------------------------------")

    return prediction                              # ✅ return value for reuse

if __name__ == "__main__":
    predict_single_engine()