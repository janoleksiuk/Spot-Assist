import argparse
import logging
import math
import sys
import time

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.api import basic_command_pb2, manipulation_api_pb2, arm_command_pb2, robot_command_pb2, synchronized_command_pb2
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
            return False
        
        traj_feedback = mobility_feedback.se2_trajectory_feedback

        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print("--- WALKING: SUCCESS ---")
            return True
        
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
            return True
        
        # Check if command failed
        elif mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("--- SITTING: FAILED ---")
            return False
            
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
            return True
        
        # Check if command failed
        elif mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("--- STANDING: FAILED ---")
            return False

        # Wait before checking again
        time.sleep(0.5)

# start rotating 
def start_rotating(client, rot_velocity, duration_sec = 2):
    print("--- ROTATING: INIT ---")

    try: 
        cmd = RobotCommandBuilder.synchro_velocity_command(0, 0, rot_velocity)
        robot_command = RobotCommandBuilder.build_synchro_command(cmd)
        client.robot_command(robot_command, end_time_secs=time.time() + duration_sec)
        return True
    except Exception as e:
        print(f"Failed to perform rotation: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())
        return False

# stop moving - including rotating
def stop_moving(client):
    print("--- MOVING: END ---")

    try: 
        cmd = RobotCommandBuilder.synchro_velocity_command(0, 0, 0)
        robot_command = RobotCommandBuilder.build_synchro_command(cmd)
        client.robot_command(robot_command, end_time_secs=time.time() + 1.0)
        return True
    except Exception as e:
        print(f"Failed to stop motion: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())
        return False

# raising arm
def raise_arm(client):
    try:
        sh0 = 0.0
        sh1 = -1.5
        el0 = 2.5
        el1 = 0.0
        wr0 = -1.5
        wr1 = 0.0

        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, time_since_reference_secs=1.0)

        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
        joint_move_cmd = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
        arm_cmd = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_cmd)
        sync_cmd = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_cmd)
        robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_cmd)
        full_cmd = RobotCommandBuilder.build_synchro_command(robot_cmd)
        print("Lifting the arm after grasping...")
        client.robot_command(full_cmd)
        time.sleep(2.0)
        print("Arm lift completed. Preparing for the delivery phase.")
        return True
    
    except Exception as e:
        print(f"Arm raising exception caught: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())
        return False

# moving froward with controlled velocity
def move_forward(client, fwd_vel, duration_sec=0.5):
    print(f"--- MOVING FORWARD with velocity {fwd_vel} : INIT ---")
    
    try:
        cmd = RobotCommandBuilder.synchro_velocity_command(fwd_vel, 0, 0)
        robot_command = RobotCommandBuilder.build_synchro_command(cmd)
        client.robot_command(robot_command, end_time_secs=time.time() + duration_sec)
        time.sleep(duration_sec)
        return True

    except Exception as e:
        print(f"Failed to move forward: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())
        return False