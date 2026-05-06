from dvc_setup import download_model
download_model()
import streamlit as st
import pandas as pd
import xgboost as xgb
import yaml
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AeroPredict AI Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MODERN DARK MODE STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 2.2rem; color: #00d4ff; }
    .stAlert { border-radius: 10px; }
    .css-1kyx7ws { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSET LOADING (CACHED) ---
@st.cache_resource
def load_assets():
    # Load parameters to find dataset name
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    dataset = params['data']['dataset']
    
    # Load the trained XGBoost Model
    model = xgb.XGBRegressor()
    model.load_model('models/model_xgb.json')
    
    # Load processed features for visualization
    df = pd.read_csv(f'data/processed/train_{dataset}_features.csv')
    return model, df, dataset

try:
    model, df, dataset = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}. Ensure you have run your DVC pipeline!")
    st.stop()

# --- 4. SIDEBAR CONTROL PANEL ---
st.sidebar.image("https://img.icons8.com/fluency/96/jet-engine.png")
st.sidebar.title("Fleet Controller")

# Engine Selection
unit_id = st.sidebar.selectbox("Select Engine Unit", options=sorted(df['unit_nr'].unique()))

# Time Travel Slider
engine_full_data = df[df['unit_nr'] == unit_id].copy()
max_cycle = int(engine_full_data['time_cycles'].max())
selected_cycle = st.sidebar.slider("Simulation Flight Cycle", 1, max_cycle, 1)

st.sidebar.markdown("---")
st.sidebar.write(f"**Dataset:** {dataset}")
st.sidebar.write(f"**Total Engines:** {len(df['unit_nr'].unique())}")

# --- 5. DATA PROCESSING & PREDICTION ---
# Filter data up to the selected cycle for plotting
plot_df = engine_full_data[engine_full_data['time_cycles'] <= selected_cycle]

# Get data for exactly the selected cycle for the metric cards
current_row = engine_full_data[engine_full_data['time_cycles'] == selected_cycle]

# Prepare features for prediction
drop_cols = ['unit_nr', 'time_cycles', 'RUL', 'regime']
X_input = current_row.drop(columns=[c for c in drop_cols if c in current_row.columns])
prediction = model.predict(X_input)[0]
actual_rul = current_row['RUL'].iloc[0]

# --- 6. MAIN DASHBOARD UI ---
st.title("✈️ AeroPredict: Predictive Maintenance")
st.subheader(f"Real-Time Health Monitoring: Engine Unit #{unit_id}")

# Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Cycle", f"{selected_cycle}")

with col2:
    st.metric("Predicted RUL", f"{prediction:.1f} Hrs", 
              delta=f"{prediction - actual_rul:.1f} vs Ground Truth", 
              delta_color="off")

with col3:
    # Logic for Health Status
    if prediction > 60:
        status = "HEALTHY"
        color = "green"
    elif prediction > 25:
        status = "MAINTENANCE REQ."
        color = "orange"
    else:
        status = "CRITICAL FAILURE"
        color = "red"
    st.markdown(f"**Status:** <span style='color:{color}; font-size:20px;'>{status}</span>", unsafe_allow_html=True)

with col4:
    health_pct = max(0, min(100, (prediction / 150) * 100))
    st.write("Engine Longevity")
    st.progress(float(health_pct / 100))

# --- 7. INTERACTIVE VISUALIZATION ---
st.markdown("---")
st.write("### Degradation Trend Analysis")

fig = go.Figure()

# Plot Actual RUL path
fig.add_trace(go.Scatter(
    x=engine_full_data['time_cycles'], 
    y=engine_full_data['RUL'],
    mode='lines',
    name='Actual Life Path',
    line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
))

# Plot AI Predictions up to current point
all_X = engine_full_data.drop(columns=[c for c in drop_cols if c in engine_full_data.columns])
all_preds = model.predict(all_X)

fig.add_trace(go.Scatter(
    x=plot_df['time_cycles'], 
    y=all_preds[:selected_cycle],
    mode='lines+markers',
    name='AI Health Estimate',
    line=dict(color='#00d4ff', width=3)
))

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Flight Cycles",
    yaxis_title="Remaining Useful Life (RUL)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20),
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# --- 8. ALERT SYSTEM ---
if status == "CRITICAL FAILURE":
    st.error(f"🚨 IMMEDIATE ACTION REQUIRED: Engine {unit_id} has surpassed safety thresholds.")
elif status == "MAINTENANCE REQ.":
    st.warning(f"⚠️ SCHEDULE INSPECTION: Engine {unit_id} showing early signs of wear.")
else:
    st.success(f"✅ NOMINAL OPERATION: Engine {unit_id} is performing within safety parameters.")