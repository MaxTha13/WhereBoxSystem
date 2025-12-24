# WBS: Where Box System

**Where Box System (WBS)** is a computer vision solution designed for autonomous logistics. It transforms a standard camera into an intelligent sensor capable of recognizing cargo, calculating its precise real-world coordinates, and transmitting this telemetry data to control systems (such as robotic arms or sorters).

## Features

* **Real-time Tracking:** High-speed object tracking using ArUco markers.
* **Coordinate Mapping:** precise calculation of physical coordinates ($X$, $Y$ in cm) and rotation angle relative to the camera.
* **Auto-Calibration:** Built-in utility to calculate the camera's focal length based on a reference marker.
* **TCP Server:** Integrated socket server (Host) to broadcast telemetry data to external clients over TCP/IP.
* **Logic Layer:** Automatic detection of box size (Small, Medium, Large) and orientation (Front, Top, Left, etc.) based on marker ID groups.
* **Data Logging:** Session-based logging of tracking events to `tracking_log.txt`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MaxTha13/WhereBoxSystem.git](https://github.com/MaxTha13/WhereBoxSystem.git)
    cd WhereBoxSystem
    ```

2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Requires `opencv-contrib-python` and `numpy`)*

## Configuration & Calibration

Before running the main tracker, you must calibrate the camera for your specific setup (camera height and resolution).

1.  Print an ArUco marker (default ID: 1).
2.  Measure the **real physical size** of the marker (in mm) and the **exact distance** from the camera to the marker (in mm).
3.  Open `calibration_config.json` (or run the script once to generate it) and update your measurements:
    ```json
    "calibration_settings": {
        "real_marker_size_mm": 40.0,
        "known_distance_mm": 460.0,
        "calibration_marker_id": 1
    }
    ```
4.  Run the calibration script:
    ```bash
    python camera_calibration.py
    ```
    Press **`c`** to save the calculated focal length (`f_pixels`) to the config file.

## Usage

Start the main tracking server:

```bash
python main.py
```

1.  Enter the **Port** number when prompted (press `Enter` for default `4444`).
2.  The system will start detecting markers.
3.  **Client Connection:** External clients can connect via TCP to your IP address.
4.  **Controls:**
    * **`q`**: Quit the application.
    * **`c`**: Clear the current log buffer.
    * **`A-Z` (Any letter)**: Send the current buffer of tracked boxes to the connected client and save to log.

## Data Protocol

The system groups markers by **Box ID** and sends data in the following string format:

```text
"TIME X_cm Y_cm ROTATION Side Size"
```

### Example output:
```text
"10:28:24.123 15.50 22.10 45.0 Front Medium"
```

### Protocol Definitions:
* **Box Sizes** (determined by Marker ID):
    * **Small**: ID 1-18 (Base ID: 1)
    * **Medium**: ID 19-36 (Base ID: 19)
    * **Large**: ID 37-54 (Base ID: 37)

* **Sides Mapping** (Relative to the Box Base ID):
    The system assigns 3 markers per side.
    * **Front**:  Base ID + 0...2  *(e.g., for Small box: 1, 2, 3)*
    * **Right**:  Base ID + 3...5  *(e.g., for Small box: 4, 5, 6)*
    * **Back**:   Base ID + 6...8
    * **Left**:   Base ID + 9...11
    * **Top**:    Base ID + 12...14
    * **Bottom**: Base ID + 15...17

