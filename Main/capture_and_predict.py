import subprocess
import time
import signal
import sys

# Global variables to track processes
processes = []

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

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

try:
    # Launch camera
    print("Launching body tracking...")
    p1 = subprocess.Popen(["python", r"C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\body-tracker\body_tracking.py"])
    processes.append(p1)

    time.sleep(3)

    # Launch predictor
    print("Launching predictor...")
    p2 = subprocess.Popen(["python", r"C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\predictor\pnn.py"])
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