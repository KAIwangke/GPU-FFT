import pandas as pd
from io import StringIO

def clean_csv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        header_index = next(i for i, line in enumerate(lines) if 'ID' in line)
        data_str = ''.join(lines[header_index:])
    return pd.read_csv(StringIO(data_str))

def main():
    # List of stages and file types
    stages = ['', 'stage0_', 'stage1_', 'stage2_', 'stage3_']
    types = ['cache', 'memtrans', 'occupancy', 'throughput']

    # DataFrame to hold all combined data
    combined_data = pd.DataFrame()

    # Iterate over each type and stage to clean and combine data
    for file_type in types:
        stage_data = pd.DataFrame()
        for stage in stages:
            file_name = f'{stage}{file_type}.csv'
            try:
                cleaned_data = clean_csv(file_name)
                cleaned_data['Stage'] = stage.strip('_') if stage else 'cufft'
                stage_data = pd.concat([stage_data, cleaned_data], ignore_index=True)
            except FileNotFoundError:
                print(f"{file_name} not found. Skipping...")
        
        # Combine all stages for the current type
        combined_data = pd.concat([combined_data, stage_data], ignore_index=True)

    # Save the combined data to a new CSV file
    combined_data.to_csv('combined_data.csv', index=False)
    print("All data has been cleaned and combined into combined_data.csv")

if __name__ == '__main__':
    main()
