# shared memory variables
DETECTED_POSE_MEMORY_NAME = "detected_pose_code_shm"
DETECTED_ACTION_MEMORY_NAME = "detected_action_code_shm"
PNN_INPUT_MEMORY_NAME = "pnn_input_code_shm"

# body tracking variables
BODY_IDX = 34 # ZED BODY_TRACKING MODEL version
CONFIDENCE_THR = 40 # confidence of body_point detection
SAVE_SESSION_DATA = False
POSES_DICT = {"[0]" : "sitting", 
                "[1]": "standing", 
                "[2]" : "sitting_1hand", 
                "[3]": "standing_1hand"}
