import pyzed.sl as sl
import cv2
import numpy as np
import pandas as pd
import sys
import signal
from datetime import datetime
from multiprocessing import shared_memory
from config.config import BODY_IDX, SAVE_SESSION_DATA, CONFIDENCE_THR, POSES_DICT


def apply_moving_mean(df, window_size):
    result_df = df.copy()
    columns_to_transform = df.columns[:-1]
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
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
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
        for i, row in df.iterrows():
            x1 = row['x1']
            y1 = row['y1']
            z1 = row['z1']
            for idx in remaining_indices:
                df.at[i, f'x{idx}'] = row[f'x{idx}'] - x1
                df.at[i, f'y{idx}'] = row[f'y{idx}'] - y1
                df.at[i, f'z{idx}'] = row[f'z{idx}'] - z1
        
    # Create mapping from old indices to new sequential indices
    old_to_new = {old: new for new, old in enumerate(sorted(remaining_indices))}
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
    df = df.rename(columns=rename_dict)
    
    # Add a 'label' column with default value 'standing' to match pnn.py syntax 
    df['label'] = 'standing'
    #filtering
    df = apply_moving_mean(df, 5)
    
    return df


#postprocessing - outputing session completed csv
def save_session(keypoints_blocks, header, postprocess=False):
    print("[--- BODY TRACKER ---]: Saving session data to csv")
    session_3d_matrix = np.vstack(keypoints_blocks)
    session_df = pd.DataFrame(session_3d_matrix, columns = header)
    if postprocess:
        session_df = process_df(df=session_df)
    session_df.to_csv(f"session_{datetime.now().date()}.csv" , index= False)


def main(argv):
    shm_detected_pose_name = argv[1]
    shm_detected_pose = shared_memory.SharedMemory(name=shm_detected_pose_name) 
    received_data_shape = (1,)
    shm_pnn_input_name = argv[2]
    shm_pnn_input = shared_memory.SharedMemory(name=shm_pnn_input_name, 
                                               size=(np.random.rand(15, 57).astype(np.float64)).nbytes)

    # process termination handler
    def cleanup(signum=None, frame=None):
        print("[--- BODY TRACKER ---]: cleaning up shared memory...")
        if SAVE_SESSION_DATA:
            save_session(keypoints_blocks=keypoint_3d_blocks, header=header)
        shm_detected_pose.close()
        shm_pnn_input.close()
        exit(0)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1

    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("[--- BODY TRACKER ---]: Camera open error: " + repr(err) + ". Exit program.")
        shm_detected_pose.close()
        shm_pnn_input.close()
        exit()

    body_params = sl.BodyTrackingParameters()
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_params.enable_tracking = True
    body_params.enable_segmentation = False
    body_params.enable_body_fitting = True
    body_params.body_format = sl.BODY_FORMAT.BODY_34
    if body_params.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    err = zed.enable_body_tracking(body_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("[--- BODY TRACKER ---]: Enable Body Tracking : "+repr(err)+". Exit program.")
        zed.close()
        shm_detected_pose.close()
        shm_pnn_input.close()
        exit()
    
    camera_info = zed.get_camera_information()
    image_width = camera_info.camera_configuration.resolution.width
    image_height = camera_info.camera_configuration.resolution.height
    cv2.namedWindow("ZED Body Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZED Body Tracking", image_width, image_height)
    
    image = sl.Mat()
    bodies = sl.Bodies()
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = CONFIDENCE_THR
    
    """
    CSV variables - initially the communication was based on csv file exchange
    currently it is based on sharing numpy arrays, but the processing pipeline
    here remained to be adjusted to csv exchange - to be changed in future.
    """
    header = []
    for l in range(34):
        header.extend([f'x{l}', f'y{l}', f'z{l}'])
    poses_dict = POSES_DICT
    
    i = 0 
    body_detected_idx = 0
    keypoint_3d_row = np.ones((BODY_IDX*3), dtype=float)
    keypoint_3d_blocks = []

    cv2.namedWindow("ZED Body Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZED Body Tracking", 900, 600)
    cv2.moveWindow("ZED Body Tracking", 0, 0)

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            err = zed.retrieve_bodies(bodies, body_runtime_param)
            img_cv = image.get_data()

            if bodies.is_new and bodies.body_list:
                for idx, body in enumerate(bodies.body_list):
                    # Get 3D keypoints and transform the to (15, 57) form
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
            if body_detected_idx == 15: 
                keypoint_3d_matrix = np.zeros((body_detected_idx,102), dtype = float)
                for k in range (0, body_detected_idx):
                    keypoint_3d_matrix[k] = keypoint_3d_array[k*102:k*102+102]

                # saving to general session dataframe
                keypoint_3d_blocks.append(keypoint_3d_matrix)
                df = pd.DataFrame(keypoint_3d_matrix, columns = header)

                df = process_df(df=df)

                # writing to shm multiprocessing buffers
                try:    
                    pnn_input_arr = df.to_numpy(dtype=np.float64)
                    shm_array = np.ndarray(pnn_input_arr.shape, dtype=pnn_input_arr.dtype, buffer=shm_pnn_input.buf)
                    shm_array[:] = pnn_input_arr[:]             
                except Exception as e:
                    print(f"[--- BODY TRACKER ---]: Error {e} while writing pnn input array.")
                
                body_detected_idx = 0      

            # Draw informative text on img
            pose_value_arr = "Undetected"
            try:
                pose_value_arr = np.ndarray(received_data_shape, dtype=np.int64, buffer=shm_detected_pose.buf)
                pose_string = poses_dict[str(pose_value_arr)]
            except Exception as e:
                print(f"[--- BODY TRACKER ---]: Error while putting pose string on img: {e}")

            cv2.putText(
                img_cv,                     # Image to draw on
                pose_string,                # Text
                (10, 300),                  # Position (x=10, y=30)
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
                shm_detected_pose.close()
                if SAVE_SESSION_DATA:
                    save_session(keypoints_blocks=keypoint_3d_blocks, header=header)
                break
        i += 1

    zed.disable_body_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main(sys.argv)
    except Exception as e:
        print(f"[--- BODY TRACKER ---]: Error {e}. Error exit.")