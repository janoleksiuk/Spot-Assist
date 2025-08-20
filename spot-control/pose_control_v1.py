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

from spot_behaviours import relative_move, sit, stand

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
                    exit_flag = sit(command_client)
                    current_behaviour = 'sitting'
                finally:
                    command_client.robot_command(RobotCommandBuilder.stop_command())
            
            #standing
            elif (pose_code == 1) and not (current_behaviour == 'standing'):
                
                try:
                    exit_flag = stand(command_client)
                    current_behaviour = 'standing'
                finally:
                    command_client.robot_command(RobotCommandBuilder.stop_command())

            #moving forward
            elif (pose_code == 3 and current_behaviour == 'standing'):
                
                try:
                    exit_flag = relative_move(0.5, 0, math.radians(0),
                                        command_client, robot_state_client, stairs=False)  
                    current_behaviour = 'walking'                 
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
