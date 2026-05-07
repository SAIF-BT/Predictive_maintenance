import os
import subprocess

def download_model():
    # Skip if model already exists
    if os.path.exists('models/model_xgb.json'):
        return                            

    token = os.environ.get('DAGSHUB_TOKEN')
    if not token:                               
        raise RuntimeError("DAGSHUB_TOKEN environment variable is not set!")

    subprocess.run([
        "dvc", "remote", "add", "-d", "origin",  
        "https://dagshub.com/SAIF-BT/Predictive_maintenance.dvc",
        "--force"
    ], check=True)                                

    subprocess.run([
        "dvc", "remote", "modify", "origin",    
        "auth", "basic"
    ], check=True)

    subprocess.run([
        "dvc", "remote", "modify", "origin",     
        "user", "SAIF-BT"
    ], check=True)

    subprocess.run([
        "dvc", "remote", "modify", "origin",   
        "password", token                    
    ], check=True)

    subprocess.run([
        "dvc", "pull",
        "models/model_xgb.json",               
        "-r", "origin"
    ], check=True)

    print("Model downloaded successfully.")