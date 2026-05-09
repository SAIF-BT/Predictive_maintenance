import os
import subprocess

def download_model():
    # Check if all required files already exist
    if (os.path.exists('models/model_xgb.json') and
        os.path.exists('data/processed/train_FD002_features.csv')):
        return

    token = os.environ.get('DAGSHUB_TOKEN')
    if not token:
        raise RuntimeError("DAGSHUB_TOKEN environment variable is not set!")

    print("Pulling required files from DVC remote...")

    subprocess.run([
        "dvc", "remote", "add", "-d", "origin",
        "https://dagshub.com/SAIF-BT/Predictive_maintenance.dvc",
        "--force"
    ], check=True)

    subprocess.run([
        "dvc", "remote", "modify", "origin", "auth", "basic"
    ], check=True)

    subprocess.run([
        "dvc", "remote", "modify", "origin", "user", "SAIF-BT"
    ], check=True)

    subprocess.run([
        "dvc", "remote", "modify", "origin", "password", token
    ], check=True)

    # Pull all files the app needs
    subprocess.run([
        "dvc", "pull",
        "models/model_xgb.json",
        "data/processed/train_FD002_features.csv",
        "-r", "origin"
    ], check=True)

    print("All files downloaded successfully.")
