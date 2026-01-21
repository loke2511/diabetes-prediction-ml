"""
Download the real Pima Indians Diabetes Dataset
"""
import urllib.request
import os

def download_pima_dataset():
    """Download the real PIMA dataset from UCI repository"""
    
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    # Column names for the PIMA dataset
    columns = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age',
        'Outcome'
    ]
    
    output_path = 'data/diabetes.csv'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    try:
        print("Downloading real PIMA Indians Diabetes Dataset...")
        print(f"URL: {url}")
        
        # Download the file
        urllib.request.urlretrieve(url, output_path)
        
        # Read and add column names
        import pandas as pd
        df = pd.read_csv(output_path, header=None, names=columns)
        
        # Save with headers
        df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset downloaded successfully to {output_path}")
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Features: {', '.join(columns[:-1])}")
        print(f"✓ Target: {columns[-1]}")
        print(f"\nDataset preview:")
        print(df.head())
        print(f"\nDataset statistics:")
        print(df.describe())
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Creating fallback dataset...")
        return False

if __name__ == '__main__':
    download_pima_dataset()
