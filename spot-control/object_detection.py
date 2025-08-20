import argparse
import sys
import time
import signal
import io

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import geometry_pb2, manipulation_api_pb2, arm_command_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client import frame_helpers

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

# detecting objects using YOLO model
def detect_objects(image_client, model, confidence=0.4, source_name='hand_color_image'):

    # retrieving image from spot camera (source name)
    response = image_client.get_image_from_sources([source_name])[0]
    img_data = response.shot.image.data
    pil_img = Image.open(io.BytesIO(img_data))
    img = np.array(pil_img)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not source_name == 'hand_color_image':
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    # determing detected objects within frame
    results = model(frame, conf=confidence, verbose=False, show=True)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls]
            detections.append({'label': label, 'conf': conf, 'bbox': (x1, y1, x2, y2)})
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return detections, frame

# computing distance from spot to object based on depth of the captured image
def compute_depth_to_object(image_client, bbox, source_name='hand_depth_in_hand_color_frame'):
    err_cnt = int(0)
    corr_cnt = int(0)
    output_depth = 0.0

    while True:
        # retrieving image from spot depth camera
        response = image_client.get_image_from_sources([source_name])[0]
        img_data = response.shot.image.data

        try:
            pil_img = Image.open(io.BytesIO(img_data))
            depth_img = np.array(pil_img)
        except UnidentifiedImageError:
            depth_img = np.frombuffer(img_data, dtype=np.uint16).reshape(
                response.shot.image.rows, response.shot.image.cols)

        if depth_img.dtype != np.uint16:
            depth_img = depth_img.astype(np.uint16)

        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        h, w = depth_img.shape
        center_x = np.clip(center_x, 0, w - 1)
        center_y = np.clip(center_y, 0, h - 1)

        kernel_size = 7
        half_k = kernel_size // 2
        neighbors = depth_img[max(0, center_y - half_k):min(h, center_y + half_k + 1),
                                max(0, center_x - half_k):min(w, center_x + half_k + 1)]
        valid_neighbors = neighbors[neighbors >= 0.01]

        if valid_neighbors.size < 5:
            #print("No valid neighborhood depth data.")
            err_cnt += 1
            if err_cnt > 50:
                print("Error: Invalid point cloud from depth source")
                return 0.0
            continue

        depth_mm = int(np.median(valid_neighbors))
        depth_meters = depth_mm / 1000.0
        output_depth += (depth_mm / 1000.0)
        corr_cnt += 1
        #print(f"Distance: {depth_meters:.2f} meters")
        
        if corr_cnt == int(25):
            return output_depth/25.0