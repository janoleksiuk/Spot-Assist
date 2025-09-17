import sys
import numpy as np
import signal
from multiprocessing import shared_memory

POSE_ENDPOINT_PATH = r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\action_code.txt'

#function retrieving detected pose code from endpoint
def get_pose(received_data_shape, shm_buffer):
    try:
        pose_value_arr = np.ndarray(received_data_shape, dtype=np.int64, buffer=shm_buffer.buf)
        return pose_value_arr[0]

    except Exception as e:
        print(f"[Detector module]: Error: {e}")

# fucntion for handling acquiried sequence
def handle_sequence(seq):
    
    # sitting - standing - sitting (covering possible classifier errors)
    if seq[-3:] == '010':
        try:
            with open(POSE_ENDPOINT_PATH, 'w') as f:
                f.write('1')
        except Exception as e:
            print(e)
            return False
        return True
    
    # standing - standing 1hand - standing - standing 1hand
    if seq[-4:] == '1313':
        try:
            with open(POSE_ENDPOINT_PATH, 'w') as f:
                f.write('2')
        except Exception as e:
            print(e)
            return False
        return True
    return False
    
def main(argv):
    # mapping onto memory segment holding pose values
    shm_detected_pose_name = argv[1]
    shm_detected_pose = shared_memory.SharedMemory(name=shm_detected_pose_name) 
    received_data_shape = (1,)

    #handling termination from parent process
    def cleanup(signum=None, frame=None):
        print("[Detector Module]: cleaning up shared memory...")
        shm_detected_pose.close()
        exit(0)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    sequence = '--' # starting seq cant be null due to sequence length
    prev_pose = None
    sequence_handled = False

    try:
        while(True):
            pose = get_pose(received_data_shape=received_data_shape, shm_buffer=shm_detected_pose)

            if pose != prev_pose:
                sequence += str(pose)
                prev_pose = pose
            
            if len(sequence) > 4:
                sequence_handled = handle_sequence(sequence)

            if sequence_handled:
                print(sequence)
                sequence = '--'
                prev_pose = None
                sequence_handled = False
    
    except KeyboardInterrupt:
        shm_detected_pose.close()

if __name__ == '__main__':
    try:
        main(sys.argv)
    except Exception as e:
        print(f"[Detector module]: Error {e}. Error exit.")



