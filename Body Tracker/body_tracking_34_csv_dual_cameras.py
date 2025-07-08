import pyzed.sl as sl
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import threading

BODY_IDX = 34
FRAMES = 400

def process_camera(camera_id):
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1
    
    # Set camera ID (0 for first camera, 1 for second camera)
    init_params.set_from_camera_id(camera_id)
    
    # Camera name for display and file naming
    camera_name = f"Camera_{camera_id}"
    
    print(f"Opening {camera_name}...")

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"{camera_name} Open : "+repr(err)+". Exit program.")
        return

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

    print(f"{camera_name}: Body tracking: Loading Module...")

    err = zed.enable_body_tracking(body_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"{camera_name} Enable Body Tracking : "+repr(err)+". Exit program.")
        zed.close()
        return
    
    # Setup for visualization
    camera_info = zed.get_camera_information()
    # Get camera resolution
    image_width = camera_info.camera_configuration.resolution.width
    image_height = camera_info.camera_configuration.resolution.height
    # Create OpenCV named window
    window_name = f"ZED Body Tracking - {camera_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, image_width, image_height)
    
    # Create image objects
    image = sl.Mat()
    
    # Body tracking objects
    bodies = sl.Bodies()
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40
    
    # Skeleton color and keypoint connections for visualization
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]  # BGR format
    
    i = 0 
    body_detected_idx = 0
    keypoint_3d_row = np.ones((BODY_IDX*3), dtype=float)
    keypoint_3d_array = np.array([])
    
    while i < FRAMES:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            
            # Retrieve bodies
            err = zed.retrieve_bodies(bodies, body_runtime_param)
            
            # Convert sl.Mat to OpenCV Mat
            img_cv = image.get_data()
            
            # Draw skeleton for each detected person
            if bodies.is_new and bodies.body_list:
                print(f"{camera_name}: {len(bodies.body_list)} Person(s) detected")
                
                # Iterate through all detected bodies
                for idx, body in enumerate(bodies.body_list):
                    # Select color based on body ID or index
                    color_idx = idx % len(colors)
                    skeleton_color = colors[color_idx]
                    
                    # Get 3D keypoints and transform them to (102,1) form
                    keypoint_3d = np.array([body.keypoint])
                    for j in range(0, BODY_IDX):
                        keypoint_3d_row[3*j:3*j + 3] = keypoint_3d[0][j]

                    # append to output 3D keypoints matrix
                    if body_detected_idx == 0:
                        keypoint_3d_array = keypoint_3d_row.copy()
                        body_detected_idx += 1
                    else:
                        keypoint_3d_array = np.append(keypoint_3d_array, keypoint_3d_row)
                        body_detected_idx += 1
            
            # Display the image
            cv2.imshow(window_name, img_cv)
            
            # Handle keyboard input
            key = cv2.waitKey(10)
            if key == 27:  # ESC key
                break
                
        i += 1
    
    # Postprocess matrix
    if body_detected_idx > 0:
        keypoint_3d_array = keypoint_3d_array.reshape(body_detected_idx, BODY_IDX*3)
        
        # Save to CSV file
        header = []
        for l in range(BODY_IDX):
            header.extend([f'x{l}', f'y{l}', f'z{l}'])

        df = pd.DataFrame(keypoint_3d_array, columns=header)
        now = datetime.now()
        time_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        csv_filename = f"{camera_name}_{time_string}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"{camera_name}: Data saved to {csv_filename}")
    else:
        print(f"{camera_name}: No body detections to save")

    # Close the camera and destroy windows
    zed.disable_body_tracking()
    zed.close()
    cv2.destroyWindow(window_name)
    print(f"{camera_name}: Closed")


def main():
    # Create threads for each camera
    camera0_thread = threading.Thread(target=process_camera, args=(0,))
    camera1_thread = threading.Thread(target=process_camera, args=(1,))
    
    # Start both threads
    camera0_thread.start()
    camera1_thread.start()
    
    # Wait for both threads to complete
    camera0_thread.join()
    camera1_thread.join()
    
    print("Both cameras have finished processing")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()