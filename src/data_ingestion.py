# src/data_ingestion.py
import os
import kaggle
import zipfile

def download_and_extract():
    dataset = 'behrad3d/nasa-cmaps'
    raw_data_path = 'data/raw'
    
    print(f"🚀 Fetching dataset: {dataset}...")
    
    # Downloads the zip file
    kaggle.api.dataset_download_files(dataset, path=raw_data_path, unzip=True)
    
    print(f"✅ Success! Raw data files are located in: {raw_data_path}")
    print(f"Files found: {os.listdir(raw_data_path)}")

if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    download_and_extract()