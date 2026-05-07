import os
import kaggle

def download_and_extract():
    dataset = 'behrad3d/nasa-cmaps'
    raw_data_path = 'data/raw'

    # Write kaggle.json from the KAGGLE_API_TOKEN environment variable
    kaggle_dir = os.path.expanduser('~/.config/kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')

    if not os.path.exists(kaggle_json):
        token = os.environ.get('KAGGLE_API_TOKEN', '')
        if not token:
            raise RuntimeError("KAGGLE_API_TOKEN is not set!")
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(kaggle_json, 'w') as f:
            f.write(f'{{"username":"SAIF-BT","key":"{token}"}}')
        os.chmod(kaggle_json, 0o600)
        print("Kaggle credentials written from token.")

    print(f"Fetching dataset: {dataset}...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=raw_data_path, unzip=True)
    print(f"Done! Files in: {raw_data_path}")
    print(f"Files found: {os.listdir(raw_data_path)}")

if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    download_and_extract()
