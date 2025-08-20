import subprocess
import time
import signal
import sys
from pathlib import Path

# Global variables to track processes
processes = []

# file direcotry creator function
def assemble_dir(str_subfolder: str) -> str:
    cwd = Path.cwd()
    output_dir = str(cwd).replace("\\launch", str_subfolder)
    print(output_dir)
    return output_dir

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
    
    print("All processes terminated")
    sys.exit(0)

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    try:
        # Launch predictor
        print("Launching predictor...")
        predictor_dir = assemble_dir(str_subfolder="\\pose-classifier\\pnn.py")
        p1 = subprocess.Popen(["python", predictor_dir])
        processes.append(p1)

        time.sleep(2)

        # Launch camera
        print("Launching body tracking...")
        tracker_dir = assemble_dir(str_subfolder="\\body-tracker\\body_tracking.py")
        p2 = subprocess.Popen(["python",tracker_dir])
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
    stri = assemble_dir(str_subfolder="\\ssss")

if __name__ == "__main__":
    main()