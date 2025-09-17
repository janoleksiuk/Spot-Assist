from launch.launch_detector import launch
import subprocess

def main():
    launch()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Launching failed: {e}")
        exit()


