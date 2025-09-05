import time
import numpy as np
import signal
from multiprocessing import shared_memory
from memory_management import DETECTED_POSE_MEMORY_NAME

POSE_ENDPOINT_PATH = r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\action_code.txt'

#function retrieving detected pose code from endpoint (.txt file)
def get_pose(received_data_shape, shm_buffer):

    try:
        pose_value_arr = np.ndarray(received_data_shape, dtype=np.int64, buffer=shm_buffer.buf)
        return pose_value_arr[0]

    except Exception as e:
        print(f"Bład: {e}")

# fucntion for handling acquiried sequence
def handle_sequence(seq):
    
    # sitting - standing - sitting (covering possible classifier errors)
    # if seq[-3:] == '010':
    if seq == 'bb010':
        try:
            with open(POSE_ENDPOINT_PATH, 'w') as f:
                f.write('1')
        except Exception as e:
            print(e)
            return False
        return True
    
    if seq == 'aa010':
        try:
            with open(POSE_ENDPOINT_PATH, 'w') as f:
                f.write('3')
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
    
def main():
    # mapping onto memory segment holding pose values
    shm = shared_memory.SharedMemory(name=DETECTED_POSE_MEMORY_NAME) # init it first !!!
    received_data_shape = (1,)

    #handling termination from parent process
    def cleanup(signum=None, frame=None):
        print("[Detector Module]: cleaning up shared memory...")
        shm.close()
        exit(0)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    sequence = 'aa' # starting seq cant be null due to sequence length
    prev_pose = None
    sequence_handled = False

    try:
        while(True):
            pose = get_pose(received_data_shape=received_data_shape, shm_buffer=shm)

            if pose != prev_pose:
                sequence += str(pose)
                prev_pose = pose
            
            if len(sequence) > 4:
                sequence_handled = handle_sequence(sequence)

            if sequence_handled:
                print(sequence)
                sequence = 'bb'
                prev_pose = None
                sequence_handled = False
    
    except KeyboardInterrupt:
        shm.close()

if __name__ == '__main__':
    main()



