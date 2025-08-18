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

from spot_behaviours import start_rotating, stop_moving, relative_move
from object_detection import detect_objects, compute_depth_to_object

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL_PATH = 'model\yolo11n.pt'
ROTATE_SPEED = 0.2
FORWARD_SPEED = 0.2
BACKWARD_SPEED = -0.3
TARGET_LABEL = 'bottle'
ROTATE_REFRESH_INTERVAL = 1.5
TARGET_DISTANCE_THRESHOLD = 1.0

task_completed = False
robot_command_client = None

# approaching desired object
def approach_object(img_client, robot_command_client, object_name, model):

    # searching for object 
    start_rotating(robot_command_client, 0.2, duration_sec=20)
    object_found = False

    while True:
        detections, frame = detect_objects(img_client, model, source_name='frontleft_fisheye_image')

        for det in detections:
            if det['label'] == object_name:
                x1, y1, x2, y2 = det['bbox']
                object_center = (x1 + x2) // 2
                frame_center = frame.shape[1] // 2
                offset_px = np.abs(object_center - frame_center)

                if offset_px < 100:
                    object_found = True
            
        if object_found:
            robot_command_client.robot_command(RobotCommandBuilder.stop_command())
            break

        time.sleep(0.1)

    time.sleep(1)

    # approaching object 
    dist = compute_depth_to_object(img_client, [x1, x2, y1, y2], source_name='frontleft_depth_in_visual_frame')

    try:
        exit_flag = relative_move(dist*0.9 , 0, 0, robot_command_client, robot_state_client, stairs=False)                   
    finally:
        robot_command_client.robot_command(RobotCommandBuilder.stop_command())

    # exiting if relative_move malfunctions - returns False as approaching failed
    if exit_flag:
        print("Approaching to object failed")
        return False
    
    print("Approaching succesful")
    return True
                    
# def raise_arm_after_grasp(robot_command_client):
#     sh0 = 0.0
#     sh1 = -1.5
#     el0 = 2.5
#     el1 = 0.0
#     wr0 = -1.5
#     wr1 = 0.0

#     traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
#         sh0, sh1, el0, el1, wr0, wr1, time_since_reference_secs=1.0)

#     arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
#     joint_move_cmd = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
#     arm_cmd = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_cmd)
#     sync_cmd = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_cmd)
#     robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_cmd)
#     full_cmd = RobotCommandBuilder.build_synchro_command(robot_cmd)
#     print("🔼 Lifting the arm after grasping...")
#     robot_command_client.robot_command(full_cmd)
#     time.sleep(2.0)
#     print("✅ Arm lift completed. Preparing for the delivery phase.")

# def move_forward(robot_command_client, forward_speed=FORWARD_SPEED, duration_sec=0.5):
#     cmd = RobotCommandBuilder.synchro_velocity_command(forward_speed, 0, 0)
#     robot_command = RobotCommandBuilder.build_synchro_command(cmd)
#     robot_command_client.robot_command(robot_command, end_time_secs=time.time() + duration_sec)
#     time.sleep(duration_sec)

# def move_backward_until_threshold(robot_command_client, image_client, model, threshold=0.6):
#     while True:
#         move_forward(robot_command_client, forward_speed=BACKWARD_SPEED, duration_sec=0.5)
#         detections, _ = detect_objects(image_client, model)
#         for det in detections:
#             if det['label'] == TARGET_LABEL:
#                 distance = compute_depth_to_object(image_client, det['bbox'])
#                 if distance >= threshold:
#                     print(f"📏 Backing up to a suitable distance: {distance:.2f} meters")
#                     return det

# def perform_grasp(manipulation_client, image_response, bbox):
#     x1, y1, x2, y2 = bbox
#     center_px_x = int((x1 + x2) / 2) - 0.9 
#     center_px_y = int((y1 + y2) / 2)

#     pick_vec = geometry_pb2.Vec2(x=center_px_x, y=center_px_y)
#     grasp = manipulation_api_pb2.PickObjectInImage(
#         pixel_xy=pick_vec,
#         transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,
#         frame_name_image_sensor=image_response.shot.frame_name_image_sensor,
#         camera_model=image_response.source.pinhole
#     )

#     grasp.grasp_params.grasp_palm_to_fingertip = 0.15
#     grasp.grasp_params.grasp_params_frame_name = frame_helpers.VISION_FRAME_NAME

#     request = manipulation_api_pb2.ManipulationApiRequest(
#         pick_object_in_image=grasp
#     )

#     print("🤖 Sending grasp request...")
#     response = manipulation_client.manipulation_api_command(request)
#     cmd_id = response.manipulation_cmd_id
#     start_time = time.time()

#     while True:
#         feedback_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
#             manipulation_cmd_id=cmd_id)
#         feedback = manipulation_client.manipulation_api_feedback_command(feedback_req)

#         state = feedback.current_state
#         state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(state)
#         print(f"⏱️ Grasp status: {state_name} ({state})", end='\r')

#         if state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
#             print("\n✅ Successfully grasped the object！")
#             return True
#         elif state in [
#                 manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
#                 manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
#                 manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP]:
#             print("\n❌ Grasp failed. The arm will retract.")
#             robot_command_client.robot_command(RobotCommandBuilder.arm_stow_command())
#             time.sleep(2)
#             return False

#         if time.time() - start_time > 15:
#             print("\n⌛ Grasp timed out. The arm will retract.")

#             try:
#                 robot_command_client.robot_command(RobotCommandBuilder.arm_stow_command(), timeout_sec=3)
#                 time.sleep(2.0)
#                 print("✅ Command sent: arm_stow_command()")
            
#             except Exception as e:
#                 print(f"⚠️ Arm retraction failed: {e}")
#                 return False

#         time.sleep(0.2)

# def search_for_person(robot_command_client, image_client, model,
#                            camera_source='frontleft_fisheye_image',
#                            depth_source='frontleft_depth_in_visual_frame',
#                            desired_distance=0.5):
#     print("🧍 Starting rotation and searching for a person...")
#     rotating = False
#     last_rotate_time = 0
#     last_align_time = 0
#     while True:
#         now = time.time()
#         detections, frame = detect_objects(image_client, model, camera_source)
#         found = False

#         for det in detections:
#             if det['label'] == 'bottle':
#                 found = True
#                 x1, y1, x2, y2 = det['bbox']
#                 x_center = (x1 + x2) // 2
#                 frame_center = frame.shape[1] // 2
#                 offset_px = x_center - frame_center
#                 now = time.time()

#                 print(f"🎯 Human detected. Offset: {offset_px} pixels")
                
#                 if abs(offset_px) > 100 and (now - last_align_time > 12):
#                     yaw_speed = 0.2 * np.sign(offset_px)  # Positive values indicate right turn, negative values indicate left turn.
#                     duration = min(abs(offset_px) / 300.0, 1.0)  # Maximum 1 second
#                     print(f"↪️ Auto-rotating to adjust orientation: yaw_speed={yaw_speed:.2f}, duration={duration:.2f}")
#                     start_rotating(robot_command_client, yaw_speed=yaw_speed, duration_sec=duration)
#                     time.sleep(duration)
#                     last_align_time = now
                
#                 stop_moving(robot_command_client)
#                 time.sleep(0.5)

#                 distance = compute_depth_to_object(image_client, det['bbox'], source_name=depth_source)
#                 if distance == 0:
#                     print("⚠️ Failed to get distance from depth data.")
#                     continue
#                 elif distance < 0.4:
#                     print(f"📏 Distance too close: {distance:.2f}m. Backing up.")
#                     move_forward(robot_command_client, forward_speed=-0.2)
#                 elif distance > 0.7:
#                     print(f"📏 Distance too far: {distance:.2f}m. Moving forward.")
#                     move_forward(robot_command_client, forward_speed=0.2)
#                 else:
#                     print(f"✅ Reached delivery distance: {distance:.2f}m")
#                     return True
#                 break

#         if not found:
#             if not rotating:
#                 print("🔄 No person detected. Starting rotation...")
#                 start_rotating(robot_command_client)
#                 rotating = True
#                 last_rotate_time = now
#             elif now - last_rotate_time > ROTATE_REFRESH_INTERVAL:
#                 start_rotating(robot_command_client)
#                 last_rotate_time = now
#         else:
#             rotating = False

#         cv2.imshow('Spot Search Person', frame)
#         if cv2.waitKey(10) == 27:
#             break

#     return False

# def deliver_object_to_person(robot_command_client, image_client, model,
    #                          camera_source='frontleft_fisheye_image'):
    # global task_completed
    # try:
    #     print("🤖 Final orientation correction...")

    #     detections, frame = detect_objects(image_client, model, camera_source)
    #     for det in detections:
    #         if det['label'] == 'person':
    #             x1, y1, x2, y2 = det['bbox']
    #             x_center = (x1 + x2) // 2
    #             frame_center = frame.shape[1] // 2
    #             offset_px = x_center - frame_center

    #             if abs(offset_px) > 50:
    #                 yaw_speed = 0.2 * np.sign(offset_px)
    #                 duration = min(abs(offset_px) / 200.0, 1.0)
    #                 print(f"↪️ Final orientation correction: yaw_speed={yaw_speed:.2f}, duration={duration:.2f}")
    #                 start_rotating(robot_command_client, yaw_speed=yaw_speed, duration_sec=duration)
    #                 time.sleep(duration)
    #                 stop_moving(robot_command_client)
    #             break  # Detect the first person

    #     print("🤲 Delivering the object...")
    #     time.sleep(5.0)
    #     robot_command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
    #     print("🖐️ Opening the gripper..")
    #     time.sleep(2.0)
    #     robot_command_client.robot_command(RobotCommandBuilder.arm_stow_command())
    #     print("📥 Stowing the arm...")
    #     time.sleep(2.0)
    #     robot_command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
    #     print("🪑 Task finished.")

    #     task_completed = True

    # except Exception as e:
    #     print(f"❌ Error occurred during delivery phase: {e}")

def main():
    
    # Initial auto-setup
    global robot_command_client, robot_state_client
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--camera-source', default='hand_color_image', help='Using camera source')
    options = parser.parse_args()

    try:
        bosdyn.client.util.setup_logging(options.verbose)
        sdk = bosdyn.client.create_standard_sdk('SpotCameraYOLO')
        robot = sdk.create_robot(options.hostname)
        bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync()

        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        image_client = robot.ensure_client(ImageClient.default_service_name)
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

        model = YOLO(MODEL_PATH)
        print(f"Loading the YOLOv11 model : {MODEL_PATH}")

        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Failed to power on Spot"
            blocking_stand(robot_command_client, timeout_sec=10)
            time.sleep(1)
            
            # approach grabbable object
            obj_approached = False
            obj_approached = approach_object(image_client, robot_command_client, object_name='bottle', model=model)
            time.sleep(1)
            
            # HERE GRAB OBJ

            # deliver object to person
            human_approached = False
            human_approached = approach_object(image_client, robot_command_client, object_name='person', model=model)

            # HERE RELEASE OBJ

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
