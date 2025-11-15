from multiprocessing import shared_memory
import numpy as np
from config.config import DETECTED_POSE_MEMORY_NAME, \
                          DETECTED_ACTION_MEMORY_NAME, \
                          PNN_INPUT_MEMORY_NAME


def init_memory_segment(name, size):
	return shared_memory.SharedMemory(create=True, size=size, name=name)


def memory_init():
	return [
		init_memory_segment(name=DETECTED_POSE_MEMORY_NAME, size=8),
        init_memory_segment(name=DETECTED_ACTION_MEMORY_NAME, size=8),
        init_memory_segment(name=PNN_INPUT_MEMORY_NAME, size=(np.random.rand(15, 57).astype(np.float64)).nbytes)
    ]    


def make_cleanup_handler(shm):
    def cleanup(signum=None, frame=None):
        print("Cleaning up shared memory...")
        shm.close()
        exit(0)
    return cleanup



