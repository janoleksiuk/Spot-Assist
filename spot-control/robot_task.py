import time
import threading
import numpy as np

from bosdyn.client.robot_command import RobotCommandBuilder, blocking_stand
from bosdyn.api import geometry_pb2, manipulation_api_pb2
from bosdyn.client import frame_helpers

from utils.spot_behaviours import start_rotating, relative_move, raise_arm, move_forward, release_gripper
from utils.object_detection import detect_objects, compute_depth_to_object


# robot behaviour variables
ROT_VEL = 0.2

# robot action variables
# define your own for your custom robot action
FIRST_TARGET = 'bottle'
SECOND_TARGET = 'person'
GRAB_OBJECT = 'bottle'


# approaching object is based on the Spot built-in cameras object detection
# for better results use SpotCam module
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

        if not detections:
            continue

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

    distance = compute_depth_to_object(img_client, [x1, x2, y1, y2], source_name='frontleft_depth_in_visual_frame') * 0.75
    try:
        exit_flag = relative_move(distance, 0, 0, robot_command_client, robot_state_client, stairs=False)
    finally:
        robot_command_client.robot_command(RobotCommandBuilder.stop_command())

    if not exit_flag:
        print("[--- SPOT CONTROL ---]: Approaching to object failed")
        return False

    return True


# grabbing object is based on the SpotArm gripper camera object detection
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


# robot action exectuded by detecting action - modify it or add your custom function using spot_behaviours
# this version commands robot to:
    # 1. Localize and approach given object
    # 2. Grab given object
    # 3. Localize and approach second object
    # 4. Deliver first given object
    # where object are defined globally (grab object = bottle, second object = person from default YOLOv11 object library)
def robot_action(robot_command_client, robot_state_client=None, image_client=None, manipulation_client=None, model=None):
    # initial stand-up
    blocking_stand(robot_command_client, timeout_sec=1)

    # find and approach bottle
    obj_approached = approach_object(robot_command_client, image_client, robot_state_client, object_name=FIRST_TARGET, model=model)

    # grab bottle
    obj_grabbed = grab_object(robot_command_client, image_client, manipulation_client, object_name=GRAB_OBJECT, model=model)

    # relocate arm
    raised_arm = raise_arm(robot_command_client)

    # find and approach human
    human_approached = approach_object(robot_command_client, image_client, robot_state_client, object_name=SECOND_TARGET, model=model)

    # release gripper
    gripper_released = release_gripper(robot_command_client)

    # move backwards
    mv_fwd = move_forward(robot_command_client, fwd_vel=-0.5, duration_sec=1)

    task_completed = obj_approached and obj_grabbed and human_approached and gripper_released and mv_fwd and raised_arm
    return task_completed
