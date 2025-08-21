import subprocess
import time
import signal
import sys
from pathlib import Path 
from memory_management import memory_init, DETECTED_POSE_MEMORY_NAME

# Global variables to track processes
processes = []

# file direcotry creator function
def assemble_dir(str_subfolder: str) -> str:
    cwd = Path.cwd()
    output_dir = str(cwd).replace("\\launch", str_subfolder)
    print(output_dir)
    return output_dir

def main():
    # Init memory segments for communication
    shms = []
    shms = memory_init()

    # signal handler function
    def signal_handler(sig, frame):
        """Handle interrupt signals by terminating child processes"""
        print("\nShutting down processes...")
        for process in processes:
            if process.poll() is None:  # Check if process is still running
                print(f"Terminating process with PID: {process.pid}")
                process.terminate()
                # Give it a moment to terminate gracefully
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print(f"Process {process.pid} did not terminate gracefully, killing...")
                    process.kill()
            
        # handle connections
        try:
            for shm in shms:
                print(f"[Launcher]: Cleaning up memeory segment {shm}")
                shm.close()
                shm.unlink()

        except Exception as e:
            print(f"Error: {e}. Did not correclty unlinked shared memory. Possible memory leakage")
        
        print("All processes terminated")
        sys.exit(0)
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    try:
        # Launch camera
        print("Launching body tracking...")
        tracker_dir = assemble_dir(str_subfolder="\\body-tracker\\body_tracking.py")
        p1 = subprocess.Popen(["python",tracker_dir])
        processes.append(p1)

        # Launch predictor
        print("Launching predictor...")
        predictor_dir = assemble_dir(str_subfolder="\\pose-classifier\\pnn.py")
        p2 = subprocess.Popen(["python", predictor_dir, DETECTED_POSE_MEMORY_NAME])
        processes.append(p2)


        print("All processes started. Press Ctrl+C to quit.")
        
        # Wait for processes to complete
        p1.wait()
        p2.wait()

    except KeyboardInterrupt:
        # This will be caught by the signal handler
        pass
    except Exception as e:
        print(f"Error occurred: {e}")
        # Clean up processes on error
        signal_handler(None, None)

    print("Exiting main program")

if __name__ == "__main__":
    main()