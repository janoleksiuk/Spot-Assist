from multiprocessing import shared_memory
import numpy as np

DETECTED_POSE_MEMORY_NAME = "detected_pose_code_shm"
DETECTED_SEQ_MEMORY_NAME = "detected_seq_code_shm"
PNN_INPUT_MEMORY_NAME = "pnn_input"

def init_memory_segment(name, size):
	return shared_memory.SharedMemory(create=True, size=size, name=name)

def memory_init():
	return [
		init_memory_segment(name=DETECTED_POSE_MEMORY_NAME, size=8)
    ]    

def make_cleanup_handler(shm):
    def cleanup(signum=None, frame=None):
        print("[Predictor Module]: cleaning up shared memory...")
        shm.close()
        exit(0)
    return cleanup

