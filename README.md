# SpotAssist

Model-based system for supporting people in daily activities using **Boston Dynamics Spot** robot and real-time human activity recognition.

![ezgif com-speed (1)](https://github.com/user-attachments/assets/23986807-7753-41b3-9512-1ff1b075a650)

---

## Overview

Project is a **framework for robotized solutions for people with limited independence due to age and mobility restrictions**. The system is composed of a human behavior monitoring module for activity recognition, and a reasoning robot that executes actions based on the classified human behavior (e.g., upon detecting that a person stands up and picks up a cup from the table, the robot will identify and deliver a bottle of water). A simplified conceptual scheme is provided below:

<img width="1400" height="638" alt="Przechwytywanie" src="https://github.com/user-attachments/assets/5a393ac9-568a-46ad-8b61-5d967ae42a7a" />

**Key Features:**
- Real-time body tracking using **ZED 2** stereo camera 
- Human activity recogntion based on [custom Probabilistic Neural Network (PNN) model](https://ieeexplore.ieee.org/document/10309359)
- Sample human activity dataset 
- **Boston Dynamics Spot** action programs using **SPOT SDK**

---

## Requirements

- **Boston Dynamics Spot** robot with **Spot Arm**
- **SPOT SDK** ([download here](https://github.com/boston-dynamics/spot-sdk))
- **ZED 2** stereo camera
- **ZED SDK 5.0+** ([download here](https://www.stereolabs.com/en-pl/developers/release))
- **CUDA 12.1+**
- **Python 3.12+**
- **Windows OS** (preferred 10/11)

 ---

 ## Usage
 
1. **Connect hardware with your PC**
- Connect **ZED 2** camera with PC using USB 3.0 port
- Connect **Boston Dynamics Spot** using Wi-Fi

2. **Install dependencies before first launch**
```bash
python -m pip install -r requirements.txt
```

3. **Launch human activity detection module**
```bash
python main.py
```

4. **Run Boston Dynamics Spot control program**
In another CLI terminal run:
```bash
cd spot-control
py action-control.py YOUR_SPOT_MODEL_IP_HERE
```

---

## Project structure
```
src/
├── body-tracker/
│   └── body_tracking.py/            # Body tracking program utilizing ZED 2 camera
├── config/                          # Launcher configuration 
├── launch/
│   └── launch_detector.py           # Tracking and detection system launch manager
│   └── memory_managment.py          # Multiprocess shared memory manager
├── pose-classifier/                 # PNN-based classifier decoding body tracking input to recognized human activities files
│   └── reference_data/              # PNN classifier reading input data helper function
│           └── reference_data.csv   # CSV file storing reference human bofy_tracking labelled data (PNN reference) 
│   └── detect_human_action.py       # human behaviours (classified multipose sequences) detector
│   └── pnn.py                       # PNN-based classifier decoding body tracking input to single pose
│   └── read_action.py               # PNN classifier reading input data helper function
├── spot-control/                    # Boston Dynamics Spot control programs and utils (also contains examples)
│   └── examples/
│           └── example_01.py        # Boston Dynamics Spot basic actions (sit/stand/moving forwad) based on detected single pose
│           └── example_02.py        # Boston Dynamics Spot basic actions (sit/stand) based on detected single action
│   └── model/
│           └── yolo11n.pt           # Ultralytics YOLO11 nano model (change for your suited YOLO model, nano recommended for high perfomance)
│   └── utils/
│           └── object_detection.py  # Function for YOLO-based object detection based on Boston Dynamics Spot cameras
│           └── shared_memory.py     # Interface for system launcher multiprocess shared memory manager
│           └── spot_behaviours.py   # Library containing safe Boston Dynamics Spot motion commands execution 
│           └── spot_utils.py        # Library for Boston Dynamics Spot utility functions
│   └── action_control.py            # Main Boston Dynamics Spot controller - adjust robot_action() function for your custom system (implemented searching and delivering bottle)
├── main.py
.
```
**Note:** To adjust system to your custom human actions data upload your `pose-classifier/reference_data/reference_data.csv` labelled reference data for your custom activities (required format: 34 BODY TRACKING ZED human keypoints data preprocessed to 19 points - to get body tracking and preprocessing code see `body-tracker/body_tracking.py` file.

