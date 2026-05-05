import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os

def plot_engine_results(unit_id=81):
    # Load model
    model = xgb.XGBRegressor()
    model.load_model('models/model_xgb.json')
    
    # Load feature data
    df = pd.read_csv('data/processed/train_FD001_features.csv')
    
    # Filter for a specific validation engine (Unit 81 is in our val set)
    engine_df = df[df['unit_nr'] == unit_id]
    X = engine_df.drop(columns=['unit_nr', 'time_cycles', 'RUL'])
    y_true = engine_df['RUL']
    
    # Predict
    y_pred = model.predict(X)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(engine_df['time_cycles'], y_true, label='Actual RUL', color='blue', linewidth=2)
    plt.plot(engine_df['time_cycles'], y_pred, label='Predicted RUL', color='red', linestyle='--')
    
    plt.title(f'Remaining Useful Life Prediction - Engine #{unit_id}')
    plt.xlabel('Time Cycles')
    plt.ylabel('RUL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/engine_81_prediction.png')
    print(f"📈 Visualization saved to plots/engine_81_prediction.png")

if __name__ == "__main__":
    plot_engine_results()