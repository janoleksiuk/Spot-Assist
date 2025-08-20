import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

FILTERING = True
PLOTTING = False
CUSTOM_NAMING = True
ACTIVITY = 'standing_1hand'

#filtering - simple moving mean
def apply_moving_mean(df, window_size):

    # Create a copy of the original DataFrame to avoid modifying it
    result_df = df.copy()
    
    # Get all column names except the last one
    columns_to_transform = df.columns[:-1]
    
    # Apply moving mean to each column except the last one
    for column in columns_to_transform:
        result_df[column] = df[column].rolling(window=window_size, min_periods=1).mean()
    
    return result_df

def apply_SG_filter(df, window_length, polyorder):
    
    # Create a copy of the original DataFrame to avoid modifying it
    result_df = df.copy()
    
    # Get all column names except the last one
    columns_to_transform = df.columns[:-1]
    
    # Apply moving mean to each column except the last one
    for column in columns_to_transform:
        result_df[column] = savgol_filter(result_df[column], window_length, polyorder)

    #plotting for debugging 
    if PLOTTING:
        plt.plot()

        ln = len(df['y0'])
        horizont_x = np.linspace(0, ln, ln)

        plt.plot(horizont_x, df['y0'], linewidth=4)
        plt.plot(horizont_x, result_df['y0'])
        plt.show()

    return result_df

#processing csv (from ZED 34 point to 19 point model)
def process_csv(csv_file, output_file=None):

    # Load the CSV file
    df = pd.read_csv(csv_file)
    df = df * (-1)
    print(f"Loaded {len(df)} rows from {csv_file}")
    
    # Delete columns related to specific keypoints
    keypoints_to_remove = [7, 9, 10, 14, 16, 17, 21, 25, 27, 28, 29, 30, 31, 32, 33]
    cols_to_drop = []
    
    for kp in keypoints_to_remove:
        cols_to_drop.extend([f'x{kp}', f'y{kp}', f'z{kp}'])
    
    # Remove any columns that don't actually exist in the DataFrame
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    # Drop the columns
    df = df.drop(columns=cols_to_drop)
    print(f"Removed keypoint columns: {cols_to_drop}")
    
    # Find remaining keypoint indices
    remaining_indices = set()
    for col in df.columns:
        if col.startswith('x'):
            try:
                idx = int(col[1:])
                if f'x{idx}' in df.columns and f'y{idx}' in df.columns and f'z{idx}' in df.columns:
                    remaining_indices.add(idx)
            except ValueError:
                pass
    
    print(f"Remaining keypoint indices: {sorted(remaining_indices)}")
    
    # Transform coordinates to make keypoint1 the origin (0,0,0)
    if 1 in remaining_indices:
        print("Transforming coordinates to make keypoint1 the origin (0,0,0)")
        
        # Process each row to transform coordinates
        for i, row in df.iterrows():
            # Get the coordinates of keypoint1
            x1 = row['x1']
            y1 = row['y1']
            z1 = row['z1']
            
            # Subtract keypoint1 coordinates from all keypoints
            for idx in remaining_indices:
                df.at[i, f'x{idx}'] = row[f'x{idx}'] - x1
                df.at[i, f'y{idx}'] = row[f'y{idx}'] - y1
                df.at[i, f'z{idx}'] = row[f'z{idx}'] - z1
        
        # Verify keypoint1 is now at origin
        print(f"Keypoint1 coordinates after transformation (should be ~0): x={df['x1'].mean()}, y={df['y1'].mean()}, z={df['z1'].mean()}")
    else:
        print("Keypoint1 not found in the data, skipping coordinate transformation")
    
    # Create mapping from old indices to new sequential indices
    old_to_new = {old: new for new, old in enumerate(sorted(remaining_indices))}
    print(f"Mapping from old to new indices: {old_to_new}")
    
    # Create a dictionary to store column renames
    rename_dict = {}
    for col in df.columns:
        if col.startswith(('x', 'y', 'z')):
            try:
                prefix = col[0]  # 'x', 'y', or 'z'
                old_idx = int(col[1:])
                if old_idx in old_to_new:
                    new_idx = old_to_new[old_idx]
                    rename_dict[col] = f'{prefix}{new_idx}'
            except ValueError:
                pass
    
    # Rename the columns
    df = df.rename(columns=rename_dict)
    print(f"Renamed columns from old to new keypoint indices")
    
    # Add a 'label' column with default value 'walking'
    df['label'] = ACTIVITY
    print(f"Added 'label' column with default value 'walking'")

    #mofify label column accordingly
    # Update the labels according to the specified ranges
    inertia = 34
    if CUSTOM_NAMING:
        df.loc[0:370 + inertia, 'label'] = 'sitting'       # Rows 0 to 999 (inclusive)
        df.loc[371+ inertia:656+ inertia, 'label'] = 'standing'  # Rows 1000 to 1399 (inclusive)
        df.loc[657+ inertia:1034+ inertia, 'label'] = 'walking'  # Rows 1400 to 1999 (inclusive)
        df.loc[1035+ inertia:1200+ inertia, 'label'] = 'standing'       # Rows 0 to 999 (inclusive)
        df.loc[1201+ inertia:1400+ inertia, 'label'] = 'walking'  # Rows 1000 to 1399 (inclusive)
        df.loc[1401+ inertia:1519+ inertia, 'label'] = 'standing'
        df.loc[1520+ inertia:1815+ inertia, 'label'] = 'sitting'       # Rows 0 to 999 (inclusive)
        df.loc[1816+ inertia:1999, 'label'] = 'standing'

    #filtering
    # for now it is hard filtered for simulation purposes
    if FILTERING:
        df = apply_moving_mean(df, 5) 
        df = apply_SG_filter(df, 55, 2)
        df = apply_moving_mean(df, 13) 
    
    # Save to file if output_file is specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Saved processed data to {output_file}")
    
    print(f"Final columns: {df.columns.tolist()}")


    return df

# Example usage:
if __name__ == "__main__":
    input_file = "data_34\34.csv"  # Replace with your input file
    output_file = "data_19\19.csv"  # Replace with your desired output file
    
    processed_df = process_csv(input_file, output_file)