import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse

def download_file(args):
    url, dest_dir = args
    filename = url.split('/')[-1]
    
    save_dir = os.path.join(dest_dir, 'downloads')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        return True
    
    # Скачиваем файл
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download data')
    parser.add_argument('data_file', type=str, help='Path to data file (TSV)')
    parser.add_argument('dest_dir', type=str, help='Directory to save images')
    args = parser.parse_args()
    
    os.makedirs(args.dest_dir, exist_ok=True)

    print("Reading data file...")
    data = pd.read_csv(args.data_file, sep='\t')
    
    download_args = [(row['downloadUrl'], args.dest_dir) for _, row in data.iterrows() if not pd.isna(row['downloadUrl'])]
    
    print("Starting download...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(
            executor.map(download_file, download_args),
            total=len(download_args),
            desc="Скачивание изображений"
        ))
    
    success = sum(results)
    total = len(results)
    print(f"\nCompleted! Successfully downloaded {success} from {total} files")
    
    if success < total:
        print(f"Failed to download {total - success} files")

if __name__ == "__main__":
    main()