import os
import subprocess

def download_model():
    # Only download if the model file is missing
    if not os.path.exists('models/model_xgb.json'):
        token = os.environ.get('DAGSHUB_TOKEN')
        # Setup DVC connection on the Streamlit server
        subprocess.run(["dvc", "remote", "add", "origin", "https://dagshub.com/SAIF-BT/Predictive_maintenance.dvc", "--force"])
        subprocess.run(["dvc", "remote", "modify", "origin", "--local", "auth", "basic"])
        subprocess.run(["dvc", "remote", "modify", "origin", "--local", "user", "SAIF-BT"])
        subprocess.run(["dvc", "remote", "modify", "origin", "--local", "password", token])
        # Pull the model
        subprocess.run(["dvc", "pull", "-r", "origin"])