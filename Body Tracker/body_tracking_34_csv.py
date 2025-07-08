import pyzed.sl as sl
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

BODY_IDX = 34
FRAMES = 2000

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
    body_runtime_param.detection_confidence_threshold = 40
    
    # Skeleton color and keypoint connections for visualization
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]  # BGR format
    
    # Define skeleton connections (keypoint pairs that form limbs)
    # skeleton_connections = [
    #     [0, 1], [1, 2], [2, 3], [3, 4],  # Right arm
    #     [0, 5], [5, 6], [6, 7],  # Left arm
    #     [0, 9], [9, 10], [10, 11],  # Right leg
    #     [0, 12], [12, 13], [13, 14],  # Left leg
    #     [0, 15], [15, 16], [16, 17],  # Spine and head
    # ]

    skeleton_connections = [
        [14, 16], [0, 14], [0, 15], [15, 17],
        [0, 1],
        [3, 4], [2, 3], [1, 2], [1, 5], [5, 6], [6, 7],
        [1, 8], [1, 11],
        [8, 9], [11, 12],
        [9, 10], [12, 13]
    ]
    
    i = 0 
    body_detected_idx = 0
    keypoint_3d_row = np.ones((BODY_IDX*3), dtype=float)
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
                print(f"{len(bodies.body_list)} Person(s) detected")
                
                # Iterate through all detected bodies
                for idx, body in enumerate(bodies.body_list):
                    # Select color based on body ID or index
                    color_idx = idx % len(colors)
                    skeleton_color = colors[color_idx]
                    
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
                    
            #         # Draw keypoints
            #         for j, kp in enumerate(keypoint_2d):
            #             if kp[0] != 0 and kp[1] != 0:  # Check if keypoint is valid
            #                 cv2.circle(img_cv, (int(kp[0]), int(kp[1])), 5, skeleton_color, -1)
                            
            #                 # Add keypoint index for debugging
            #                 cv2.putText(img_cv, str(j), (int(kp[0]), int(kp[1])), 
            #                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
            #         # Draw connections between keypoints
            #         for connection in skeleton_connections:
            #             start_idx = connection[0]
            #             end_idx = connection[1]
                        
            #             start_point = keypoint_2d[start_idx]
            #             end_point = keypoint_2d[end_idx]
                        
            #             # Check if both keypoints are valid
            #             if (start_point[0] != 0 and start_point[1] != 0 and 
            #                 end_point[0] != 0 and end_point[1] != 0):
            #                 cv2.line(img_cv, 
            #                         (int(start_point[0]), int(start_point[1])), 
            #                         (int(end_point[0]), int(end_point[1])), 
            #                         skeleton_color, 2)
                    
            #         # Display body ID and confidence
            #         if body_params.enable_tracking:
            #             id_text = f"ID: {int(body.id)} Conf: {int(body.confidence)}%"
            #             position = body.position
            #             pos_text = f"Pos: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]m"
                        
            #             # Get position for text (use the neck keypoint if available)
            #             text_x, text_y = int(keypoint_2d[0][0]), int(keypoint_2d[0][1]) - 20
                        
            #             cv2.putText(img_cv, id_text, (text_x, text_y), 
            #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, skeleton_color, 2)
            #             cv2.putText(img_cv, pos_text, (text_x, text_y + 20), 
            #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, skeleton_color, 2)
            
            # # Add frame counter
            # cv2.putText(img_cv, f"Frame: {i+1}/100", (30, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the image
            cv2.imshow("ZED Body Tracking", img_cv)
            
            # Handle keyboard input
            key = cv2.waitKey(10)
            if key == 27:  # ESC key
                break
                
        i += 1
        
    #postprocess matrix
    keypoint_3d_matrix = np.zeros((body_detected_idx,102), dtype = float)
    for k in range (0, body_detected_idx):
        keypoint_3d_matrix[k] = keypoint_3d_array[k*102:k*102+102]

    #save to csv file
    header = []
    for l in range(34):
        header.extend([f'x{l}', f'y{l}', f'z{l}'])

    df = pd.DataFrame(keypoint_3d_matrix, columns = header)
    now = datetime.now()
    time_string = time_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    df.to_csv(time_string + '.csv', index = False)
    # df.to_csv('34.csv', index= False)

    # Close the camera and destroy windows
    zed.disable_body_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()