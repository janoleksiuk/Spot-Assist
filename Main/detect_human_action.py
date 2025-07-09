import time

POSE_ENTRYPOINT_PATH = r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\behaviour_code.txt'
POSE_ENDPOINT_PATH = r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\action_code.txt'

#function retrieving detected pose code from endpoint (.txt file)
def get_pose():

    while(True):
        try:
            with open(POSE_ENTRYPOINT_PATH, 'r') as f:
                endpoint_content = f.read().strip()  
                if endpoint_content == '':
                    continue
                detected_pose = int(endpoint_content)
                return detected_pose
        except FileNotFoundError:
            print("Endpoint file not found.")
            continue

# fucntion for handling acquiried sequence
def handle_sequence(seq):
    
    # sitting - standing - sitting
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
    
def main():

    sequence = 'a' # starting seq cant be null due to sequence length
    prev_pose = None
    sequence_handled = False

    while(True):
        pose = get_pose()

        if pose != prev_pose:
            sequence += str(pose)
            prev_pose = pose
        
        if len(sequence) > 3:
            sequence_handled = handle_sequence(sequence)

        if sequence_handled:
            sequence = 'a'
            prev_pose = None
            sequence_handled = False

if __name__ == '__main__':
    main()



