import subprocess
import time
import signal
import sys
from pathlib import Path 
from launch.memory_management import memory_init, DETECTED_POSE_MEMORY_NAME, \
                                                  DETECTED_ACTION_MEMORY_NAME, \
                                                  PNN_INPUT_MEMORY_NAME

# file direcotry creator function
def assemble_dir(*parts: str) -> str:
    project_root = Path(__file__).resolve().parent.parent  # Go up from launch/
    full_path = project_root.joinpath(*parts)
    return str(full_path)


def launch():
    # Init memory segments for communication
    shms = []
    shms = memory_init()
    processes = []

    # process termination handler
    def signal_handler(sig, frame):
        print("[--- LAUNCHER ---]: Shutting down processes.")
        for process in processes:
            if process.poll() is None:  
                print(f"[--- LAUNCHER ---]: Terminating process with PID: {process.pid}")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print(f"[--- LAUNCHER ---]: Process {process.pid} did not terminate gracefully, killing...")
                    process.kill()
        try:
            for shm in shms:
                print(f"[--- LAUNCHER ---]: Cleaning up memeory segment {shm}")
                shm.close()
                shm.unlink()
        except Exception as e:
            print(f"[--- LAUNCHER ---]: Error: {e}. Did not correclty unlinked shared memory. Possible memory leakage")
        
        print("[--- LAUNCHER ---]: All processes terminated")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    try:
        print("[--- LAUNCHER ---]: Launching BODY TRACKING module.")
        tracker_dir = assemble_dir("body-tracker", "body_tracking.py")
        p1 = subprocess.Popen(["python", tracker_dir, DETECTED_POSE_MEMORY_NAME,
                                                      PNN_INPUT_MEMORY_NAME])
        processes.append(p1)

        # Launch pose classifier
        print("[--- LAUNCHER ---]: Launching CLASSIFIER module.")
        classifier_dir = assemble_dir("pose-classifier", "pnn.py")
        p2 = subprocess.Popen(["python", classifier_dir, DETECTED_POSE_MEMORY_NAME,
                                                         PNN_INPUT_MEMORY_NAME])
        processes.append(p2)

        # Launch action detector
        print("[--- LAUNCHER ---]: Launching ACTION DETECTOR module.")
        detector_dir = assemble_dir("pose-classifier", "detect_human_action.py")
        p3 = subprocess.Popen(["python", detector_dir, DETECTED_POSE_MEMORY_NAME, 
                                                       DETECTED_ACTION_MEMORY_NAME])
        processes.append(p3)

        print("[--- LAUNCHER ---]: All processes started. Press Ctrl+C to quit.")
        p1.wait()
        p2.wait()
        p3.wait()

    except Exception as e:
        print(f"[--- LAUNCHER ---]: Error occurred: {e}")
        signal_handler(None, None)

