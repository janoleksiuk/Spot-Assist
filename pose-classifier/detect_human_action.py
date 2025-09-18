import sys
import numpy as np
import signal
from multiprocessing import shared_memory


def get_pose(received_data_shape, shm_buffer):
    try:
        pose_value_arr = np.ndarray(received_data_shape, dtype=np.int64, buffer=shm_buffer.buf)
        return pose_value_arr[0]
    except Exception as e:
        print(f"[--- ACTION DETECTOR ---]: Error while getting pose value: {e}")


def write_action(value, shm_buffer):
    try:
        shm_buffer.buf[:8] = value.to_bytes(8, byteorder='little', signed=True)
    except Exception as e:
        print(f"[--- ACTION DETECTOR ---]: Error while writing action code: {e}")


def handle_sequence(seq, shm_buffer):
    # sitting - standing - sitting 
    if seq[-3:] == '010':
        write_action(value=1, shm_buffer=shm_buffer)
        return True
    
    # standing - standing 1hand - standing - standing 1hand
    if seq[-4:] == '1313':
        write_action(value=2, shm_buffer=shm_buffer)
        return True
    """
    Append with your other combinations here
    """    
    return False
    

def main(argv):
    # mapping onto memory segment holding pose and action values
    shm_detected_pose_name = argv[1]
    shm_detected_pose = shared_memory.SharedMemory(name=shm_detected_pose_name) 
    shm_detected_pose_received_data_shape = (1,)
    shm_detected_action_name = argv[2]
    shm_detected_action = shared_memory.SharedMemory(name=shm_detected_action_name) 

    # process termination handling
    def cleanup(signum=None, frame=None):
        print("[--- ACTION DETECTOR ---]: cleaning up shared memory...")
        shm_detected_pose.close()
        shm_detected_action.close()
        exit(0)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    sequence = '--' # starting seq cant be null due to sequence length
    prev_pose = None
    sequence_handled = False

    while(True):
        pose = get_pose(received_data_shape=shm_detected_pose_received_data_shape, shm_buffer=shm_detected_pose)

        if pose != prev_pose:
            sequence += str(pose)
            prev_pose = pose
        
        if len(sequence) > 4:
            sequence_handled = handle_sequence(seq=sequence, shm_buffer=shm_detected_action)

        if sequence_handled:
            sequence = '--'
            prev_pose = None
            sequence_handled = False
    

if __name__ == '__main__':
    try:
        main(sys.argv)
    except Exception as e:
        print(f"[--- ACTION DETECTOR ---]: Error: {e}. Error exit.")



