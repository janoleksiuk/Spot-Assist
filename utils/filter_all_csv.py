import os
import pandas as pd
import glob

def apply_moving_mean(df, window_size=5):

    # Create a copy of the original DataFrame to avoid modifying it
    result_df = df.copy()
    
    # Get all column names except the last one
    columns_to_transform = df.columns[:-1]
    
    # Apply moving mean to each column except the last one
    for column in columns_to_transform:
        result_df[column] = df[column].rolling(window=window_size, min_periods=1).mean()
    
    return result_df

def process_csv_files(folder_path, output_folder=None, window_size=3):

    # If output folder is not specified, use the same folder as input
    if output_folder is None:
        output_folder = folder_path
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    # Process each CSV file
    for file_path in csv_files:
        try:
            # Get the filename without the path
            filename = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(filename)[0]
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Apply moving mean
            processed_df = apply_moving_mean(df, window_size)
            
            # Create output file path
            output_file = os.path.join(output_folder, f"{file_name_without_ext}_processed.csv")
            
            # Save processed DataFrame to CSV
            processed_df.to_csv(output_file, index=False)
            
            print(f"Processed {filename} -> {os.path.basename(output_file)}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    # You can modify these parameters as needed
    folder_path = r"pre-csv"  # Change this to your folder path
    output_folder =  r"post-csv"   # Optional, change this or set to None
    window_size = 5  # Change the window size as needed
    
    process_csv_files(folder_path, output_folder, window_size)