import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient

# printing battery level
def print_battery_level(robot_state):
    print("BATTERY STATE: " + str(robot_state.battery_states))