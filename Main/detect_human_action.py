POSE_ENDPOINT_PATH = r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\behaviour_code.txt'

#function retrieving detected pose code from endpoint (.txt file)
def get_pose():

    while(True):
        try:
            with open(POSE_ENDPOINT_PATH, 'r') as f:
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
        print("SEQ 010 SPOTTED")
        return True
    
    else:
        return False
    

def main():

    sequence = ''
    prev_pose = None
    sequence_handled = False

    while(True):
        pose = get_pose()
        print(sequence)

        if pose != prev_pose:
            sequence += str(pose)
            prev_pose = pose
        
        if len(sequence) > 2:
            sequence_handled = handle_sequence(sequence)

        if sequence_handled:
            sequence = ''
            prev_pose = None
            sequence_handled = False

if __name__ == '__main__':
    main()



