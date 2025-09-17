import argparse
import logging
import numpy as np
import sys
import time
import traceback
import signal
from multiprocessing import shared_memory

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

from utils.spot_behaviours import sit, stand
from utils.spot_utils import print_battery_state
from utils.shared_memory import DETECTED_ACTION_MEMORY_NAME

#function retrieving detected action code from endpoint
def get_action(received_data_shape, shm_buffer):
    try:
        action_value_arr = np.ndarray(received_data_shape, dtype=np.int64, buffer=shm_buffer.buf)
        return action_value_arr[0]
    except Exception as e:
        print(f"[SPOT CONTROL]: Error while getting pose value: {e}")
        return 0

# function freezing - wait t second
def countdown(t):
    i = 0   
    while(i<t+1):
        print(t - i)
        i = i + 1
        time.sleep(1)

# main program
def run(config):
    shm_detected_action = shared_memory.SharedMemory(name=DETECTED_ACTION_MEMORY_NAME) 
    shm_detected_action_received_data_shape = (1,)

    def cleanup(signum=None, frame=None):
        print("[SPOT CONTROL]: cleaning up shared memory...")
        shm_detected_action.close()
        exit(0)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
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
        print_battery_state(state)
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
        prev_action = 0

        # Time to get ready for operator on the scene
        countdown(5)

        # Main loop
        while(True):

            action_code = get_action(received_data_shape=shm_detected_action_received_data_shape, shm_buffer=shm_detected_action)  
            
            #behaviour for sequence sit -> stand -> sit
            if (action_code == 1) and (prev_action != 1):
                
                try:
                    exit_flag = not stand(command_client)
                    prev_action = 1
                finally:
                    command_client.robot_command(RobotCommandBuilder.stop_command())

            #behaviour for sequence stand -> stand_1h -> stand -> stand_1h
            if (action_code == 2) and (prev_action != 2):
                
                try:
                    exit_flag = not sit(command_client)
                finally:
                    command_client.robot_command(RobotCommandBuilder.stop_command())

            if exit_flag:
                print("--- EXIT ---")
                shm_detected_action.close()
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
