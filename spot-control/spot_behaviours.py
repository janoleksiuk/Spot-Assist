import argparse
import logging
import math
import sys
import time

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

# walking within relative frame - robot reaches x,y location within its reference frame - returns True if fails
def relative_move(dx, dy, dyaw, robot_command_client, robot_state_client, frame_name=ODOM_FRAME_NAME, stairs=False):
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
            return True
        
        traj_feedback = mobility_feedback.se2_trajectory_feedback

        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print("--- WALKING: SUCCESS ---")
            return False
        
        time.sleep(1)

# sitting (all 4 legs bended) - return True if failes
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
            return False
        
        # Check if command failed
        elif mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("--- SITTING: FAILED ---")
            return True
            
        # Wait before checking again
        time.sleep(0.5)

# standing - return True if failes
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
            return False
        
        # Check if command failed
        elif mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("--- STANDING: FAILED ---")
            return True

        # Wait before checking again
        time.sleep(0.5)

# start rotating 
def start_rotating(client, rot_velocity, duration_sec = 2):
    
    print("--- ROTATING: INIT ---")

    try: 
        cmd = RobotCommandBuilder.synchro_velocity_command(0, 0, rot_velocity)
        robot_command = RobotCommandBuilder.build_synchro_command(cmd)
        client.robot_command(robot_command, end_time_secs=time.time() + duration_sec)
    except Exception as e:
        print(f"Failed to perform rotation: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())

# stop moving - including rotating
def stop_moving(client):

    print("--- MOVING: END ---")

    try: 
        cmd = RobotCommandBuilder.synchro_velocity_command(0, 0, 0)
        robot_command = RobotCommandBuilder.build_synchro_command(cmd)
        client.robot_command(robot_command, end_time_secs=time.time() + 1.0)
    except Exception as e:
        print(f"Failed to stop rotation: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())


