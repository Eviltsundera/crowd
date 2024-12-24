import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

def create_directory_structure(base_dir):
    dirs = {
        os.path.join(base_dir, split, class_name)
        for split in ['train', 'val']
        for class_name in ['conifer', 'deciduous']
    }
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def balance_classes(df):
    class_counts = df['is_conifer'].value_counts()
    min_class_count = class_counts.min()
    
    balanced_df = pd.concat([
        df[df['is_conifer'] == True].sample(min_class_count),
        df[df['is_conifer'] == False].sample(min_class_count)
    ])
    
    return balanced_df.sample(frac=1).reset_index(drop=True)

def split_and_copy_files(df, source_dir, dest_dir, test_size=0.2):
    df = balance_classes(df)
    
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['is_conifer'],
        random_state=42
    )
    
    create_directory_structure(dest_dir)
    
    def copy_files(data, split):
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f'Copying {split} files'):
            filename = row['downloadUrl'].split('/')[-1]
            source_path = os.path.join(source_dir, 'downloads', filename)
            
            if not os.path.exists(source_path):
                continue
                
            dest_subdir = 'conifer' if row['is_conifer'] else 'deciduous'
            destination_path = os.path.join(dest_dir, split, dest_subdir, filename)
            
            shutil.copy2(source_path, destination_path)
    
    copy_files(train_df, 'train')
    copy_files(val_df, 'val')
    
    print("\nDataset statistics:")
    print(f"Total images: {len(df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print("\nClass distribution:")
    print("Train:")
    print(train_df['is_conifer'].value_counts())
    print("\nValidation:")
    print(val_df['is_conifer'].value_counts())

def main():
    parser = argparse.ArgumentParser(description='Split data into train and validation sets')
    parser.add_argument('data_file', type=str, help='Path to data file (TSV)')
    parser.add_argument('source_dir', type=str, help='Directory with downloaded images')
    parser.add_argument('dest_dir', type=str, help='Directory for train/val split')
    parser.add_argument('--test-size', type=float, default=0.2, help='Validation set size (default: 0.2)')
    
    args = parser.parse_args()
    
    print("Reading data file...")
    data = pd.read_csv(args.data_file, sep='\t')
    
    split_and_copy_files(data, args.source_dir, args.dest_dir, args.test_size)

if __name__ == "__main__":
    main()
