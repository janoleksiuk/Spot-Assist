import pandas as pd

def combine_csv_files():
    """
    Combine three specific CSV files into a single CSV file.
    The input files are hardcoded as:
    - 09042025152643.csv
    - 09042025152752.csv
    - 09042025152834.csv
    
    The output will be saved as 'combined_output.csv'
    """
    # Hardcoded input file paths
    input_files = [
        'a (1).csv',
        'a (2).csv',
        'a (3).csv',
        'a (4).csv',
        'a (5).csv',
        'a (6).csv',
        'a (7).csv',
        'a (8).csv',
        'a (9).csv',
        'a (10).csv',
        'a (11).csv',
        'a (12).csv',
        'a (13).csv',
        'a (14).csv'
    ]

    # input_files =[
    #     'a (1).csv',
    #     'a (2).csv'
    # ]
    
    # Hardcoded output file path
    output_file = 'combined_output.csv'
    
    print(f"Combining {len(input_files)} CSV files...")
    
    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()
    
    # Process each file
    for file_path in input_files:
        try:
            print(f"Processing: {file_path}")
            # Read the current CSV file
            df = pd.read_csv(file_path)
            
            # Append the data to the combined DataFrame
            combined_data = pd.concat([combined_data, df], ignore_index=True)
            
            print(f"  - Added {len(df)} rows from {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Shuffle the rows
    print(f"Shuffling {len(combined_data)} rows...")
    combined_data = combined_data.sample(frac=1).reset_index(drop=True)
    
    # Save the combined data to a new CSV file
    combined_data.to_csv(output_file, index=False)
    
    print(f"Successfully combined {len(input_files)} files.")
    print(f"Total rows: {len(combined_data)}")
    print(f"Output saved to: {output_file}")

# Run the function when the script is executed
if __name__ == "__main__":
    combine_csv_files()