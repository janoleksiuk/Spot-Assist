import pyzed.sl as sl
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

BODY_IDX = 34
CONFIDENCE_THR = 40
FREQ = 2

# FPS 30 #

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

#preprocessing - rearagning acquired input from 34 raw 3d keypoints data to 19 filtered 3d keypoints data with default label
def process_df(df):

    #rotating by 180 degree (since the input is inversed)
    df = df * (-1)
    
    # Delete columns related to specific keypoints
    keypoints_to_remove = [7, 9, 10, 14, 16, 17, 21, 25, 27, 28, 29, 30, 31, 32, 33]
    cols_to_drop = []
    
    for kp in keypoints_to_remove:
        cols_to_drop.extend([f'x{kp}', f'y{kp}', f'z{kp}'])
    
    # Remove any columns that don't actually exist in the DataFrame
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    # Drop the columns
    df = df.drop(columns=cols_to_drop)
    
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
    
    # Transform coordinates to make keypoint1 the origin (0,0,0)
    if 1 in remaining_indices:
        
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
        
    # Create mapping from old indices to new sequential indices
    old_to_new = {old: new for new, old in enumerate(sorted(remaining_indices))}
    
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
    
    # Add a 'label' column with default value 'standing' to match pnn.py syntax
    df['label'] = 'standing'

    #filtering
    df = apply_moving_mean(df, 5)
    
    return df

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    body_params = sl.BodyTrackingParameters()
    # Different model can be chosen, optimizing the runtime or the accuracy
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_params.enable_tracking = True
    body_params.enable_segmentation = False
    # Optimize the person joints position, requires more computations
    body_params.enable_body_fitting = True
    body_params.body_format = sl.BODY_FORMAT.BODY_34

    if body_params.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        # positional_tracking_param.set_as_static = True
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Body tracking: Loading Module...")

    err = zed.enable_body_tracking(body_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Enable Body Tracking : "+repr(err)+". Exit program.")
        zed.close()
        exit()
    
    # Setup for visualization
    camera_info = zed.get_camera_information()
    # Get camera resolution
    image_width = camera_info.camera_configuration.resolution.width
    image_height = camera_info.camera_configuration.resolution.height
    # Create OpenCV named window
    cv2.namedWindow("ZED Body Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZED Body Tracking", image_width, image_height)
    
    # Create image objects
    image = sl.Mat()
    
    # Body tracking objects
    bodies = sl.Bodies()
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = CONFIDENCE_THR
    
    #csv variables
    header = []
    for l in range(34):
        header.extend([f'x{l}', f'y{l}', f'z{l}'])
    
    #initializing variables
    i = 0 
    body_detected_idx = 0
    keypoint_3d_row = np.ones((BODY_IDX*3), dtype=float)

    #create frame
    # Create the window with a name
    cv2.namedWindow("ZED Body Tracking", cv2.WINDOW_NORMAL)

    # Set fixed size: width=300, height=400
    cv2.resizeWindow("ZED Body Tracking", 900, 600)

    # Move window to top-left corner of screen (x=0, y=0)
    cv2.moveWindow("ZED Body Tracking", 0, 0)

    #body tracking
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            
            # Retrieve bodies
            err = zed.retrieve_bodies(bodies, body_runtime_param)
            
            # Convert sl.Mat to OpenCV Mat
            img_cv = image.get_data()
            
            # Draw skeleton for each detected person
            if bodies.is_new and bodies.body_list:
                # print(f"{len(bodies.body_list)} Person(s) detected")
                
                # Iterate through all detected bodies
                for idx, body in enumerate(bodies.body_list):
                    
                    # Get 3D keypoints and transform the to (102,1) form
                    keypoint_3d = np.array([body.keypoint])
                    for j in range (0, BODY_IDX):
                        keypoint_3d_row[3*j:3*j + 3] = keypoint_3d[0][j]

                    # append to output 3D keypoints matrix
                    if body_detected_idx == 0:
                        keypoint_3d_array = keypoint_3d_row
                        body_detected_idx += 1
                    else:
                        keypoint_3d_array = np.append(keypoint_3d_array, keypoint_3d_row, axis=0)
                        body_detected_idx +=1
                    
            # create dataframe every 15 frames - to be used by predictor:
            if body_detected_idx == int(30/FREQ):

                #arrange 102x10 matrix from 1x1020 array  
                keypoint_3d_matrix = np.zeros((body_detected_idx,102), dtype = float)
                for k in range (0, body_detected_idx):
                    keypoint_3d_matrix[k] = keypoint_3d_array[k*102:k*102+102]
                
                df = pd.DataFrame(keypoint_3d_matrix, columns = header)

                # preprocess data
                df = process_df(df=df)

                # save df as csv in prod directory
                df.to_csv(r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\19.csv', index= False)

                #reset variables
                body_detected_idx = 0      

            # Draw informative text on img
            try:
                with open(r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\pose_string.txt', 'r') as f:
                    pose_string = f.read().strip()  # strip removes any newlines or spaces
                    if pose_string == '':
                        raise ValueError("File is empty")
            except FileNotFoundError:
                pose_string = 'Undetected'
            except ValueError as e:
                pose_string = 'Undetected'

            cv2.putText(
                img_cv,                     # Image to draw on
                pose_string,                 # Text
                (10, 300),                   # Position (x=10, y=30)
                cv2.FONT_HERSHEY_SIMPLEX,   # Font
                5,                          # Font scale
                (0, 0, 255),                # Color (Green in BGR)
                5,                          # Thickness
                cv2.LINE_AA                 # Line type for anti-aliasing
            )    

            # Display the image
            cv2.imshow("ZED Body Tracking", img_cv)
            
            # Handle keyboard input
            key = cv2.waitKey(10)
            if key == 27:  # ESC key
                break
                
        i += 1

    # Close the camera and destroy windows
    zed.disable_body_tracking()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()