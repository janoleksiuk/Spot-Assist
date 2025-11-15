import time
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import  arm_command_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    get_se2_a_tform_b,
)


# walking within relative frame - robot reaches x,y location within its reference frame - returns True if fails
def relative_move(dx, dy, dyaw, robot_command_client, robot_state_client, frame_name=ODOM_FRAME_NAME, stairs=False):
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified frame. The command will stop at the new pos
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
    end_time = 10.0
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)
    
    # Waiting until the robot has reached the goal.
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("[--- SPOT CONTROL ---]: Walking failed.")
            return False
        
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            return True
        
        time.sleep(1)


# sitting (all 4 legs bended)
def sit(client):
    cmd = RobotCommandBuilder.synchro_sit_command()
    end_time = 5.0
    cmd_id = client.robot_command(lease=None, command=cmd,
                                           end_time_secs=time.time() + end_time)

    while True:
        feedback = client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        sit_feedback = mobility_feedback.sit_feedback
        
        if sit_feedback.status == sit_feedback.STATUS_IS_SITTING:
            return True
        elif mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("[--- SPOT CONTROL ---]: Sitting failed.")
            return False

        time.sleep(0.5)


def stand(client):
    cmd = RobotCommandBuilder.synchro_stand_command()
    end_time = 5.0
    cmd_id = client.robot_command(lease=None, command=cmd,
                                           end_time_secs=time.time() + end_time)

    while True:
        feedback = client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        stand_feedback = mobility_feedback.stand_feedback

        if stand_feedback.status == stand_feedback.STATUS_IS_STANDING:
            return True
        elif mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("[--- SPOT CONTROL ---]: Standing failed.")
            return False

        time.sleep(0.5)


def start_rotating(client, rot_velocity, duration_sec = 2):
    try: 
        cmd = RobotCommandBuilder.synchro_velocity_command(0, 0, rot_velocity)
        robot_command = RobotCommandBuilder.build_synchro_command(cmd)
        client.robot_command(robot_command, end_time_secs=time.time() + duration_sec)
        return True
    except Exception as e:
        print(f"[--- SPOT CONTROL ---]: Failed to perform rotation: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())
        return False


def stop_moving(client):
    try: 
        cmd = RobotCommandBuilder.synchro_velocity_command(0, 0, 0)
        robot_command = RobotCommandBuilder.build_synchro_command(cmd)
        client.robot_command(robot_command, end_time_secs=time.time() + 1.0)
        return True
    except Exception as e:
        print(f"[--- SPOT CONTROL ---]: Failed to stop motion: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())
        return False


# sh0, sh1, el0, el1, wr0, wr1 - desired joint coordinates of a 6-DOF SpotARM
def raise_arm(client, sh0=0.0, sh1=-1.5, el0=2.5, el1=0.0, wr0=-1.5, wr1=0.0):
    try:
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, time_since_reference_secs=1.0)

        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
        joint_move_cmd = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
        arm_cmd = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_cmd)
        sync_cmd = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_cmd)
        robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_cmd)
        full_cmd = RobotCommandBuilder.build_synchro_command(robot_cmd)
        client.robot_command(full_cmd)
        time.sleep(2.0)
        return True

    except Exception as e:
        print(f"[--- SPOT CONTROL ---]: Arm raising exception caught: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())
        return False


# moving froward with controlled velocity
def move_forward(client, fwd_vel, duration_sec=0.5):
    try:
        cmd = RobotCommandBuilder.synchro_velocity_command(fwd_vel, 0, 0)
        robot_command = RobotCommandBuilder.build_synchro_command(cmd)
        client.robot_command(robot_command, end_time_secs=time.time() + duration_sec)
        time.sleep(duration_sec)
        return True

    except Exception as e:
        print(f"[--- SPOT CONTROL ---]: Failed to move forward: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())
        return False
    

def stow_arm(client):
    try: 
        cmd = RobotCommandBuilder.arm_stow_command()
        client.robot_command(cmd)
        return True
    except Exception as e:
        print(f"[--- SPOT CONTROL ---]: Failed to stow arm: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())
        return False
   
    
def release_gripper(client):
    try: 
        cmd = RobotCommandBuilder.claw_gripper_open_command()
        client.robot_command(cmd)
        return True
    except Exception as e:
        print(f"[--- SPOT CONTROL ---]: Failed to release gripper: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())
        return False
    

def lock_gripper(client):
    try: 
        cmd = RobotCommandBuilder.claw_gripper_close_command()
        client.robot_command(cmd)
        return True
    except Exception as e:
        print(f"[--- SPOT CONTROL ---]: Failed to lock gripper: {e}")
        client.robot_command(RobotCommandBuilder.stop_command())
        return False
