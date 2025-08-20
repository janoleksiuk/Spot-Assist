import argparse
import sys
import time
import signal
import io
import threading

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import geometry_pb2, manipulation_api_pb2, arm_command_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client import frame_helpers

from spot_behaviours import start_rotating, stop_moving, relative_move, raise_arm, move_forward
from object_detection import detect_objects, compute_depth_to_object

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL_PATH = r"model\yolo11n.pt"
ROT_VEL = 0.2
FORWARD_VEL = 0.2
FIRST_TARGET = 'suitcase'
SECOND_TARGET = 'person'
GRAB_OBJECT = 'bottle'

task_completed = False
robot_command_client = None

# approaching desired object
def approach_object(img_client, robot_command_client, object_name, model, dist=0):
    object_found = False
    stop_rotation_thread = threading.Event()
    
    # spot rotating thread
    def rotation_thread_target(robot_cmd_client, rot_vel, duration):
        while not stop_rotation_thread.is_set():
            start_rotating(robot_cmd_client, rot_vel, duration)
            time.sleep(duration)

    rotation_thread = threading.Thread(target=rotation_thread_target, args=(robot_command_client, ROT_VEL, 0.5))
    rotation_thread.start()

    while True:
        detections, frame = detect_objects(img_client, model, source_name='frontleft_fisheye_image')

        for det in detections:
            if det['label'] == object_name:
                x1, y1, x2, y2 = det['bbox']
                object_center = (x1 + x2) // 2
                frame_center = frame.shape[1] // 2
                offset_px = np.abs(object_center - frame_center)
                print(offset_px)
                
                # 15 - precise depth measurement - needs correction / 200 - forward trajectory
                if offset_px < 200:

                    object_found = True
            
        if object_found:
            stop_rotation_thread.set()
            print("Object Found")
            break

    time.sleep(0.5)
    rotation_thread.join()

    distance = dist
    # approaching object - below lines used not in demo due to precision issues
    # distance = compute_depth_to_object(img_client, [x1, x2, y1, y2], source_name='frontleft_depth_in_visual_frame')*0.67
    # print(f"DIST: {dist}")

    # correcting position - had to reach 15 px to measure distance - straight trajecotry is around 200 px
    # start_rotating(robot_command_client, -ROT_VEL, duration_sec=3)
    # time.sleep(3)

    try:
        exit_flag = relative_move(distance , 0, 0, robot_command_client, robot_state_client, stairs=False)                   
    finally:
        robot_command_client.robot_command(RobotCommandBuilder.stop_command())

    # exiting if relative_move malfunctions - returns False as approaching failed
    if exit_flag:
        print("Approaching to object failed")
        return False
    
    print("Approaching succesful")
    return True

# grabbing desired object
def grab_object(img_client, manipulation_client, object_name, model):
    object_grabbed = False
    object_detected = False

    while not object_detected:
        detections, frame = detect_objects(img_client, model)

        if len(detections) > 0:
            for det in detections:
                if det['label'] == object_name:
                    x1, y1, x2, y2 = det['bbox']
                    object_detected = True

    center_px_x = int((x1 + x2) / 2) - 0.9 
    center_px_y = int((y1 + y2) / 2)

    pick_vec = geometry_pb2.Vec2(x=center_px_x, y=center_px_y)
    image_response = img_client.get_image_from_sources(['hand_color_image'])[0]
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,
        frame_name_image_sensor=image_response.shot.frame_name_image_sensor,
        camera_model=image_response.source.pinhole
    )

    grasp.grasp_params.grasp_palm_to_fingertip = 0.15
    grasp.grasp_params.grasp_params_frame_name = frame_helpers.VISION_FRAME_NAME

    request = manipulation_api_pb2.ManipulationApiRequest(
        pick_object_in_image=grasp
    )

    print("Sending grasp request...")
    response = manipulation_client.manipulation_api_command(request)
    cmd_id = response.manipulation_cmd_id
    start_time = time.time()

    while True:
        feedback_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_id)
        feedback = manipulation_client.manipulation_api_feedback_command(feedback_req)

        state = feedback.current_state
        state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(state)
        print(f"Grasp status: {state_name} ({state})", end='\r')

        if state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
            print("Successfully grasped the object")
            return True
        
        elif state in [
                manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
                manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP]:
            print("WGrasp failed. The arm will retract.")
            robot_command_client.robot_command(RobotCommandBuilder.arm_stow_command())
            time.sleep(2)
            return False

        if time.time() - start_time > 15:
            print("Grasp timed out. The arm will retract.")

            try:
                robot_command_client.robot_command(RobotCommandBuilder.arm_stow_command(), timeout_sec=3)
                time.sleep(2.0)
                print("Command sent: arm_stow_command()")
            
            except Exception as e:
                print(f"Arm retraction failed: {e}")
                return False

        time.sleep(0.2)

def main():
    # Initial auto-setup
    global robot_command_client, robot_state_client

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--camera-source', default='hand_color_image', help='Using camera source')
    options = parser.parse_args()

    try:
        bosdyn.client.util.setup_logging(options.verbose)
        sdk = bosdyn.client.create_standard_sdk('SpotAssist')
        robot = sdk.create_robot(options.hostname)
        bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync()

        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        image_client = robot.ensure_client(ImageClient.default_service_name)
        manipulation_client = robot.ensure_client(ManipulationApiClient.default_service_name)
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

        model = YOLO(MODEL_PATH)
        print(f"Loading the YOLOv11 model : {MODEL_PATH}")

        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Failed to power on Spot"
            blocking_stand(robot_command_client, timeout_sec=10)
            time.sleep(1)
            
            #approach grabbable object
            obj_approached = False
            obj_approached = approach_object(image_client, robot_command_client, object_name=FIRST_TARGET, model=model, dist=1)
            time.sleep(1)
            
            # HERE GRAB OBJ
            obj_grabbed = False
            obj_grabbed = grab_object(image_client, manipulation_client, object_name=GRAB_OBJECT, model=model)
            time.sleep(1)

            # # deliver object to person
            human_approached = False
            human_approached = approach_object(image_client, robot_command_client, object_name=SECOND_TARGET, model=model, dist=2)
            time.sleep(1)
            
            # HERE RELEASE OBJ
            raise_arm(robot_command_client)
            time.sleep(2)

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        try:
            if not task_completed and robot and robot.is_powered_on():
                stop_moving(robot_command_client)
                robot_command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
                time.sleep(2.0)
                robot_command_client.robot_command(RobotCommandBuilder.arm_stow_command())
                time.sleep(2.0)
                robot_command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
                time.sleep(3.0)

        except Exception as e:
            print(f"Shutdown failed: {e}")

        cv2.destroyAllWindows()
        print("Spot operation completed. Exiting.")

if __name__ == '__main__':
    main()
