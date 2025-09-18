import argparse
import math
import sys
import time
import traceback
from multiprocessing import shared_memory
import numpy as np
import signal

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient)

from utils.spot_behaviours import relative_move, sit, stand
from utils.spot_utils import print_battery_level
from utils.shared_memory import DETECTED_POSE_MEMORY_NAME


def get_pose(received_data_shape, shm_buffer):
    try:
        pose_value_arr = np.ndarray(received_data_shape, dtype=np.int64, buffer=shm_buffer.buf)
        return pose_value_arr[0]
    except Exception as e:
        print(f"[sPOT CONTROL]: Error while getting pose value: {e}")
        return 0


def countdown(t):
    i = 0   
    while(i<t+1):
        print(t - i)
        i = i + 1
        time.sleep(1)


def run(config):
    shm_detected_pose = shared_memory.SharedMemory(name=DETECTED_POSE_MEMORY_NAME) 
    shm_detected_pose_received_data_shape = (1,)

    def cleanup(signum=None, frame=None):
        print("[SPOT CONTROL]: cleaning up shared memory...")
        shm_detected_pose.close()
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

        print_battery_level(state)

        # Spot power on
        robot.time_sync.wait_for_sync()
        robot.power_on()
        assert robot.is_powered_on(), '[--- SPOT CONTROL ---]: Robot power on failed.'

        exit_flag = False
        current_behaviour = ''

        # Time to get ready for operator on the scene
        countdown(5)

        # lauching system with sitting_1hand
        while(True):
            if get_pose(received_data_shape=shm_detected_pose_received_data_shape, shm_buffer=shm_detected_pose) == 2:
                break

        # Main loop
        while(True):
            pose_code = get_pose(received_data_shape=shm_detected_pose_received_data_shape, shm_buffer=shm_detected_pose) 
            # 0 - sitting; 1 - standing;  2- sitting_1hand; 3 - standing_1hand

            if (pose_code == 2) and not (current_behaviour == ''):
                exit_flag = True

            # executing behaviour
            # sitting
            elif (pose_code == 0) and not (current_behaviour == 'sitting' or current_behaviour == ''):
                try:
                    exit_flag = not sit(command_client)
                    current_behaviour = 'sitting'
                finally:
                    command_client.robot_command(RobotCommandBuilder.stop_command())
            #standing
            elif (pose_code == 1) and not (current_behaviour == 'standing'):
                
                try:
                    exit_flag = not stand(command_client)
                    current_behaviour = 'standing'
                finally:
                    command_client.robot_command(RobotCommandBuilder.stop_command())
            #moving forward
            elif (pose_code == 3 and current_behaviour == 'standing'):
                
                try:
                    exit_flag = not relative_move(0.5, 0, math.radians(0),
                                        command_client, robot_state_client, stairs=False)  
                    current_behaviour = 'walking'                 
                finally:
                    command_client.robot_command(RobotCommandBuilder.stop_command())

            if exit_flag:
                shm_detected_pose.close()
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
        logger.error('[--- SPOT CONTROL ---]: Threw an exception: %s\n%s', exc, traceback.format_exc())
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)
