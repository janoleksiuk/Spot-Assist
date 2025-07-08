import argparse
import logging
import math
import sys
import time
import traceback

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.api import basic_command_pb2
from bosdyn.api import geometry_pb2 as geo
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus

from bosdyn.client import frame_helpers, math_helpers, robot_command
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_se2_a_tform_b,
)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_for_trajectory_cmd,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient

POSE_ENDPOINT_PATH = r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\behaviour_code.txt'

#function retrieving detected pose code from endpoint (.txt file)
def get_pose():

    while(True):
        try:
            with open(POSE_ENDPOINT_PATH, 'r') as f:
                endpoint_content = f.read().strip()  # strip removes any newlines or spaces
                if endpoint_content == '':
                    continue
                detected_pose = int(endpoint_content)
                return detected_pose
        except FileNotFoundError:
            print("Endpoint file not found.")
            continue

#function for moving in the frame
def relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client, stairs=False):
    print("--- WALKING: INIT ---")
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Build the transform for where we want the robot to be relative to where the body currently is.
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    # We do not want to command this goal in body frame because the body will move, thus shifting
    # our goal. Instead, we transform this offset to get the goal position in the output frame
    # (which will be either odom or vision).
    out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified frame. The command will stop at the
    # new position.
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
    end_time = 10.0
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)
    
    # Wait until the robot has reached the goal.
    while True:
        
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("--- WALKING: FAILED ---")
            return True, ''
        
        traj_feedback = mobility_feedback.se2_trajectory_feedback

        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print("--- WALKING: SUCCESS ---")
            return False, 'walking_forward'
        
        time.sleep(1)


# sitting behaviour(all 4 legs bended)
def sit(client):
    
    print("--- SITTING: INIT ---")

    cmd = RobotCommandBuilder.synchro_sit_command()
    end_time = 5.0
    cmd_id = client.robot_command(lease=None, command=cmd,
                                           end_time_secs=time.time() + end_time)

    # ensure proper execution
    while True:
        feedback = client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        sit_feedback = mobility_feedback.sit_feedback
        
        # Check if command succeeded
        if sit_feedback.status == sit_feedback.STATUS_IS_SITTING:
            print("--- SITTING: SUCCESS ---")
            return False, 'sitting'
        
        # Check if command failed
        elif mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("--- SITTING: FAILED ---")
            return True, ''
            
        # Wait before checking again
        time.sleep(0.5)

# standing behaviour 
def stand(client):
    
    print("--- STANDING: INIT ---")

    cmd = RobotCommandBuilder.synchro_stand_command()
    end_time = 5.0
    cmd_id = client.robot_command(lease=None, command=cmd,
                                           end_time_secs=time.time() + end_time)

    # ensure proper execution
    while True:
        feedback = client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        stand_feedback = mobility_feedback.stand_feedback

        # Check if command succeeded
        if stand_feedback.status == stand_feedback.STATUS_IS_STANDING:
            print("--- STANDING: SUCCESS ---")
            return False, 'standing'
        
        # Check if command failed
        elif mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("--- STANDING: FAILED ---")
            return True, ''
        

            
        # Wait before checking again
        time.sleep(0.5)

# function freezing - wait t second
def countdown(t):

    i = 0   
    while(i<t+1):
        print(t - i)
        i = i + 1
        time.sleep(1)

# main program
def run(config):

    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('StanceClient')

    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    # Acquire lease
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        state = robot_state_client.get_robot_state()

        # acknowledge with battery state and press a to continue
        print("BATTERY STATE: " + str(state.battery_states))
        print("PRESS 'a' to proceed.")
        while True:
            user_input = input("Input: ")
            if user_input.lower() == "a":
                break

        # Power On
        robot.time_sync.wait_for_sync()
        robot.power_on()
        assert robot.is_powered_on(), 'Robot power on failed.'

        exit_flag = False
        current_behaviour = ''

        # Time to get ready for operator on the scene
        countdown(5)

        # lauching system with sitting_1hand
        while(True):
            if get_pose() == 2:
                print("--- ROBOT READY --- ")
                break

        # Main loop
        while(True):

            pose_code = get_pose() # 0 - sitting; 1 - standing;  2- sitting_1hand; 3 - standing_1hand

            if (pose_code == 2) and not (current_behaviour == ''):
                exit_flag = True

            #executing behaviour
            # sitting
            elif (pose_code == 0) and not (current_behaviour == 'sitting' or current_behaviour == ''):
                
                try:
                    exit_flag, current_behaviour = sit(command_client)
                finally:
                    command_client.robot_command(RobotCommandBuilder.stop_command())
            
            #standing
            elif (pose_code == 1) and not (current_behaviour == 'standing'):
                
                try:
                    exit_flag, current_behaviour = stand(command_client)
                finally:
                    command_client.robot_command(RobotCommandBuilder.stop_command())

            #moving forward
            elif (pose_code == 3 and current_behaviour == 'standing'):
                
                # ADD ATAN ANGLE SERVICE!
                try:
                    exit_flag, current_behaviour = relative_move(0.5, 0, math.radians(0), ODOM_FRAME_NAME,
                                        command_client, robot_state_client, stairs=False)                   
                finally:
                    command_client.robot_command(RobotCommandBuilder.stop_command())

            if exit_flag:
                print("--- EXIT ---")
                break

        robot.power_off(cut_immediately=False, timeout_sec=20)
                
def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args()

    try:
        run(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.error('Threw an exception: %s\n%s', exc, traceback.format_exc())
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)
