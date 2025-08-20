from multiprocessing import shared_memory
import numpy as np

DETECTED_POSE_MEMORY_NAME = "detected_pose_code_shm"
DETECTED_SEQ_MEMORY_NAME = "detected_seq_code_shm"
PNN_INPUT_MEMORY_NAME = "pnn_input"


# def init_memory_segments():
