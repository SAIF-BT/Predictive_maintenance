import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import yaml

def plot_results():
    # 1. Load config
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    dataset = params['data']['dataset']

    # ✅ BUG 1 fixed — create plots directory before saving anything
    os.makedirs('plots', exist_ok=True)

    # 2. Load model and data with existence checks
    model_path = 'models/model_xgb.json'
    data_path = f'data/processed/train_{dataset}_features.csv'

    if not os.path.exists(model_path):                    # ✅ BUG 2 fixed
        raise FileNotFoundError(
            f"{model_path} not found. Run the train stage first."
        )
    if not os.path.exists(data_path):                     # ✅ BUG 2 fixed
        raise FileNotFoundError(
            f"{data_path} not found. Run the features stage first."
        )

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    df = pd.read_csv(data_path)

    # 3. Pick first available engine
    unit_id = df['unit_nr'].unique()[0]
    engine_df = df[df['unit_nr'] == unit_id]

    drop_cols = ['unit_nr', 'time_cycles', 'RUL', 'regime']
    X = engine_df.drop(columns=[c for c in drop_cols if c in engine_df.columns])
    y_true = engine_df['RUL']
    y_pred = model.predict(X)

    # 4. RUL Prediction Plot
    plt.figure(figsize=(10, 5))
    plt.plot(engine_df['time_cycles'], y_true, label='Actual RUL', color='blue')
    plt.plot(engine_df['time_cycles'], y_pred, label='Predicted RUL',
             color='red', linestyle='--')
    plt.title(f'{dataset} - Engine #{unit_id} RUL Prediction')
    plt.xlabel('Flight Cycles')
    plt.ylabel('Remaining Useful Life')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/engine_prediction.png')
    plt.close()                                           # ✅ BUG 3 fixed — close figure

    # 5. Feature Importance Plot
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    feature_importance_df = feature_importance_df\
        .sort_values(by='Importance', ascending=False)\
        .head(15)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'],
             feature_importance_df['Importance'], color='teal')
    plt.xlabel('Importance Score')
    plt.title(f'{dataset} - Top 15 Most Influential Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()                                           # ✅ BUG 3 fixed — close figure

    print(f"Results visualized for {dataset} (Engine #{unit_id})")

if __name__ == "__main__":
    plot_results()