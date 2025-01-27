import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert Excel file with URLs to txt format')
    parser.add_argument('input_file', type=str, help='Path to input Excel file')
    parser.add_argument('output_file', type=str, help='Path to output txt file')
    args = parser.parse_args()
    
    df = pd.read_excel(args.input_file)
    urls = df['downloadUrl'].dropna().tolist()
    
    with open(args.output_file, 'w') as f:
        f.write("downloadUrl\n")
        for url in urls:
            f.write(f"{url}\n")
    
    print(f"Converted {len(urls)} URLs from {args.input_file} to {args.output_file}")

if __name__ == '__main__':
    main() 