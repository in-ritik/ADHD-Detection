import pandas as pd
import os

# Paths & Output
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

OUTPUT_DIR = os.path.join(DATA_DIR, "patient_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, "valid_patients_processed.csv")

def split_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Source file {INPUT_FILE} missing.")
        return

    print(f"Reading source data...")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Generating {len(df)} individual patient files...")
    
    for _, row in df.iterrows():
        patient_id = row['ID']
        patient_df = pd.DataFrame([row])
        
        filename = f"patient_{patient_id}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        patient_df.to_csv(filepath, index=False)
        
    print(f"Success. Files saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    split_data()
