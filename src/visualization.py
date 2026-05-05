import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import yaml

def plot_results():
    # 1. Load config to find the right dataset
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    dataset = params['data']['dataset']
    
    # 2. Load model and the correct data
    model = xgb.XGBRegressor()
    model.load_model('models/model_xgb.json')
    
    data_path = f'data/processed/train_{dataset}_features.csv'
    df = pd.read_csv(data_path)
    
    # 3. Generate RUL Prediction Plot for a sample engine
    # FD002 has more engines, let's pick one that exists (e.g., unit_nr 81)
    unit_id = df['unit_nr'].unique()[0] # Automatically picks the first available engine
    engine_df = df[df['unit_nr'] == unit_id]
    
    # Drop non-feature columns
    drop_cols = ['unit_nr', 'time_cycles', 'RUL', 'regime']
    X = engine_df.drop(columns=[c for c in drop_cols if c in engine_df.columns])
    y_true = engine_df['RUL']
    y_pred = model.predict(X)
    
    plt.figure(figsize=(10, 5))
    plt.plot(engine_df['time_cycles'], y_true, label='Actual RUL', color='blue')
    plt.plot(engine_df['time_cycles'], y_pred, label='Predicted RUL', color='red', linestyle='--')
    plt.title(f'{dataset} - Engine #{unit_id} RUL Prediction')
    plt.legend()
    plt.savefig('plots/engine_prediction.png')
    plt.close()

    # 4. Generate Feature Importance Plot
    importances = model.feature_importances_
    feat_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(15)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='teal')
    plt.xlabel('Importance Score')
    plt.title(f'{dataset} - Top 15 Most Influential Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig('plots/feature_importance.png')
    print(f"📈 Results visualized for {dataset} (Sample Engine #{unit_id})")

if __name__ == "__main__":
    plot_results()