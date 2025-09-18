import argparse
import sys
import time
import signal
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from multiprocessing import shared_memory

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import geometry_pb2, manipulation_api_pb2
from bosdyn.client import frame_helpers

from utils.spot_behaviours import start_rotating, stop_moving, relative_move, raise_arm, move_forward
from utils.object_detection import detect_objects, compute_depth_to_object
from utils.spot_utils import print_battery_level
from utils.shared_memory import DETECTED_ACTION_MEMORY_NAME


MODEL_PATH = r"model\yolo11n.pt"
ROT_VEL = 0.2
FORWARD_VEL = 0.2
FIRST_TARGET = 'bottle'
SECOND_TARGET = 'person'
GRAB_OBJECT = 'bottle'


def approach_object(robot_command_client, img_client, robot_state_client, object_name, model, dist=0):
    object_found = False
    stop_rotation_thread = threading.Event()
    
    # spot rotating thread
    def rotation_thread_target(robot_cmd_client, rot_vel, duration):
        while not stop_rotation_thread.is_set():
            start_rotating(robot_cmd_client, rot_vel, duration)
            time.sleep(duration)

    rotation_thread = threading.Thread(target=rotation_thread_target, args=(robot_command_client, -ROT_VEL, 0.5))
    rotation_thread.start()
    
    while True:
        detections, frame = detect_objects(img_client, model, source_name='frontright_fisheye_image')

        for det in detections:
            if det['label'] == object_name:
                x1, y1, x2, y2 = det['bbox']
                object_center = (x1 + x2) // 2
                frame_center = frame.shape[1] // 2
                offset_px = np.abs(object_center - frame_center)
                print(offset_px)
                
                # 15 - precise depth measurement, but unstable
                px_thr = 100
                if offset_px < px_thr:
                    object_found = True
            
        if object_found:
            stop_rotation_thread.set()
            print("Object Found")
            break

    time.sleep(0.5)
    rotation_thread.join()

    distance = compute_depth_to_object(img_client, [x1, x2, y1, y2], source_name='frontleft_depth_in_visual_frame')*0.75
    try:
        exit_flag = relative_move(distance , 0, 0, robot_command_client, robot_state_client, stairs=False)                   
    finally:
        robot_command_client.robot_command(RobotCommandBuilder.stop_command())

    if not exit_flag:
        print("[--- SPOT CONTROL ---]: Approaching to object failed")
        return False
    
    return True


def grab_object(robot_command_client, img_client, manipulation_client, object_name, model):
    object_detected = False
    while not object_detected:
        detections, _ = detect_objects(img_client, model)
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

    response = manipulation_client.manipulation_api_command(request)
    cmd_id = response.manipulation_cmd_id
    start_time = time.time()

    while True:
        feedback_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_id)
        feedback = manipulation_client.manipulation_api_feedback_command(feedback_req)

        state = feedback.current_state
        if state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
            return True
        
        elif state in [
                manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
                manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP]:
            print("[--- SPOT CONTROL ---]: Grasp failed. The arm will retract.")
            robot_command_client.robot_command(RobotCommandBuilder.arm_stow_command())
            time.sleep(2)
            return False

        if time.time() - start_time > 15:
            print("[--- SPOT CONTROL ---]: Grasp timed out. The arm will retract.")
            try:
                robot_command_client.robot_command(RobotCommandBuilder.arm_stow_command(), timeout_sec=3)
                time.sleep(2.0)
                print("Command sent: arm_stow_command()")
            except Exception as e:
                print(f"Arm retraction failed: {e}")
                return False

        time.sleep(0.2)


def get_action(received_data_shape, shm_buffer):
    try:
        action_value_arr = np.ndarray(received_data_shape, dtype=np.int64, buffer=shm_buffer.buf)
        return action_value_arr[0]
    except Exception as e:
        print(f"[--- SPOT CONTROL ---]: Error while getting pose value: {e}")
        return 0


def main():
    # mapping multiprocessing shared memory
    shm_detected_action = shared_memory.SharedMemory(name=DETECTED_ACTION_MEMORY_NAME) 
    shm_detected_action_received_data_shape = (1,)

    # handling process termination 
    def cleanup(signum=None, frame=None):
        print("[SPOT CONTROL]: cleaning up shared memory...")
        shm_detected_action.close()
        exit(0)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    bosdyn.client.util.setup_logging(options.verbose)
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--camera-source', default='hand_color_image', help='Using camera source')
    options = parser.parse_args()

    try:
        task_completed = False
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
        
        state = robot_state_client.get_robot_state()
        print_battery_level(state)

        model = YOLO(MODEL_PATH)

        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "[--- SPOT CONTROL ---]: Failed to power on Spot"

            # waiting for appropiate pose
            while True:
                if not get_action(received_data_shape=shm_detected_action_received_data_shape, shm_buffer=shm_detected_action) == 1: 
                # code action 1 = sit -> stand -> sit
                    time.sleep(0.5)
                else:
                    # initial stand-up
                    blocking_stand(robot_command_client, timeout_sec=10)
                    time.sleep(1)

                    # find and approach bottle
                    obj_approached = approach_object(robot_command_client, image_client, robot_state_client, object_name=FIRST_TARGET, model=model)
                    time.sleep(1)

                    # grab bottle
                    obj_grabbed = grab_object(robot_command_client, image_client, manipulation_client, object_name=GRAB_OBJECT, model=model)
                    time.sleep(1)

                    # relocate arm
                    raise_arm(robot_command_client)
                    time.sleep(1)

                    # find and approach human
                    human_approached = approach_object(robot_command_client, image_client, robot_state_client, object_name=SECOND_TARGET, model=model)
                    time.sleep(2)

                    # release gripper
                    robot_command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
                    time.sleep(1)

                    move_forward(robot_command_client, fwd_vel=-0.5, duration_sec=1)
                    task_completed = obj_approached and obj_grabbed and human_approached
                    break

    except Exception as e:
        print(f"[--- SPOT CONTROL ---]: An exception occurred: {e}")

    finally:
        try:
            if not task_completed:
                print("[--- SPOT CONTROL ---]: Task not accomplished.")
                stop_moving(robot_command_client)
                time.sleep(1)
                robot_command_client.robot_command(RobotCommandBuilder.arm_stow_command())
                time.sleep(1)
                robot_command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
                time.sleep(1)
                robot.power_off(cut_immediately=False, timeout_sec=20)
            else:
               stop_moving(robot_command_client)
               time.sleep(1)
               robot_command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
               time.sleep(1) 
               robot.power_off(cut_immediately=False, timeout_sec=20)

        except Exception as e:
            print(f"[--- SPOT CONTROL ---]: Shutdown failed: {e}")

        cv2.destroyAllWindows()
        shm_detected_action.close()
        

if __name__ == '__main__':
    if not main():
        sys.exit(1)
