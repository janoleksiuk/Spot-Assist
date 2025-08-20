# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Test script to run a simple stance command.
"""
import argparse
import sys
import time
import traceback

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client import frame_helpers, math_helpers, robot_command
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient

def run(config):
    """Testing API Stance

    This example will cause the robot to power on, stand and reposition its feet (Stance) at the
    location it's already standing at.

    * Use sw-estop running on tablet/python etc.
    * Have ~1m of free space all around the robot
    * Ctrl-C to exit and return lease.
    """

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


        # This example ues the current body position, but you can specify any position.
        # A common use is to specify it relative to something you know, like a fiducial.
        vo_T_body = frame_helpers.get_se2_a_tform_b(state.kinematic_state.transforms_snapshot,
                                                    frame_helpers.VISION_FRAME_NAME,
                                                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)

        # Power On
        robot.power_on()
        assert robot.is_powered_on(), 'Robot power on failed.'
        
        # Stand
        robot_command.blocking_stand(command_client)
        print('starting - wait 3 seconds.')
        time.sleep(3)

        # Acquiring actual x,y stance offsets
        print('Robot standing - acquiring initial stance parameters...')
        # get_actual_stance_offsets(robot_state_client.get_robot_state())
        
        iter = 0
        pose = 'standing'
        exit_flag = False

        while(True):

            #retrieving pose id
            while(True):
                user_input = input("pose id: ")
                if user_input.lower() == '1':
                    pose = 'sitting'
                    break
                elif user_input.lower() == '2':
                    pose = 'standing'
                    break
                elif user_input.lower() == '0':
                    exit_flag = True
                    break
                else:
                    print('Incorrect pose id')
            
            #executing behaviours
            if pose == 'sitting':
                print('...sitting...')
                cmd = RobotCommandBuilder.synchro_sit_command()
                command_client.robot_command(cmd)
                time.sleep(5)

            elif pose == 'standing':
                print('...standing...')
                cmd = RobotCommandBuilder.synchro_stand_command()
                command_client.robot_command(cmd)
                time.sleep(5)

            if exit_flag:
                break
            
        robot.power_off(cut_immediately=False, timeout_sec=20)
                
def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--x-offset', default=0.3, type=float, help='Offset in X for Spot to step')
    parser.add_argument('--y-offset', default=0.3, type=float, help='Offset in Y for Spot to step')
    options = parser.parse_args()

    if not 0.2 <= abs(options.x_offset) <= 0.5:
        print('Invalid x-offset value. Please pass a value between 0.2 and 0.5')
        sys.exit(1)
    if not 0.1 <= abs(options.y_offset) <= 0.4:
        print('Invalid y-offset value. Please pass a value between 0.1 and 0.4')
        sys.exit(1)

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
