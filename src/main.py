import cv2
import numpy as np
import cv2.aruco as aruco
import os
import json
import math
import socket
import threading
from datetime import datetime
from collections import deque
import time

# --- SERVER GLOBAL SETTINGS ---
# IP "0.0.0.0" allows the server to accept connections from any network interface
SERVER_IP = "0.0.0.0"
SERVER_PORT = 4444
active_client_conn = None  # Stores the current active socket connection
server_running = True      # Flag to control the background server thread


def ask_for_settings():
    """
    Prompts the user to define the server port on startup.
    Defaults to 4444 if input is empty.
    """
    global SERVER_PORT
    print("\n" + "=" * 40)
    print("    SERVER MODE (HOST)")
    print("=" * 40)
    print(f"This machine will wait for incoming connections.")
    print(f"Listening IP: {SERVER_IP} (All interfaces)")

    port_input = input(f"Enter Port (Enter for {SERVER_PORT}): ").strip()
    if port_input.isdigit():
        SERVER_PORT = int(port_input)

    print(f"[SETUP] Server will listen on port: {SERVER_PORT}")
    print("=" * 40 + "\n")


def server_thread_func():
    """
    Background thread function to handle TCP server operations.
    It listens for connections and accepts clients without blocking the main video loop.
    """
    global active_client_conn, server_running

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allows the port to be reused immediately after the program closes
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.listen(1)  # Listen for 1 connection at a time
        print(f"[SERVER] Waiting for connection on port {SERVER_PORT}...")

        while server_running:
            if active_client_conn is None:
                try:
                    server_socket.settimeout(1.0)  # Timeout to check 'server_running' periodically
                    conn, addr = server_socket.accept()
                    print(f"\n[SERVER] >>> CLIENT CONNECTED: {addr} <<<")
                    active_client_conn = conn
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[SERVER] Accept error: {e}")
            else:
                # If a client is connected, idle and let the main thread send data
                time.sleep(0.1)

    except Exception as e:
        print(f"[SERVER] Critical server error: {e}")
    finally:
        server_socket.close()


def send_data_to_client(message):
    """
    Sends a string message to the connected client.
    Handles potential disconnection errors gracefully.
    """
    global active_client_conn

    if active_client_conn:
        try:
            active_client_conn.sendall(message.encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            print("\n[SERVER] Client disconnected during transmission.")
            active_client_conn.close()
            active_client_conn = None
        except Exception as e:
            print(f"[SERVER] Send error: {e}")


# --- LOGGING AND OBJECT SETTINGS ---
LOG_FILE = "tracking_log.txt"
LOG_STRING = ""
SMOOTHING_WINDOW = 5  # Number of frames for moving average smoothing
SIDE_NAMES = ["Front", "Right", "Back", "Left", "Top", "Bottom"]
BOX_SIZES = {1: "Small", 2: "Medium", 3: "Large"}  # Map Box IDs to Size labels


def initialize_log_file():
    """Creates or appends to the log file and writes the session header."""
    global LOG_FILE
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        if file_exists:
            f.write("\n\n" + "=" * 60 + "\n")
        else:
            f.write("=" * 60 + "\n")
        f.write(f"NEW SESSION: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write('Format: "TIME X Y ROTATION SIDE SIZE" (string)\n')
        f.write("=" * 60 + "\n\n")
    return LOG_FILE


def write_to_log(message, also_print=True):
    """Writes a message to the file and optionally prints it to console."""
    global LOG_FILE, LOG_STRING
    LOG_STRING += message + "\n"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + "\n")
    if also_print: print(message)


def get_log_string(): return LOG_STRING


def clear_log_string(): 
    global LOG_STRING
    LOG_STRING = ""
    print("[CLEAR] In-memory log cleared")


def format_log_string(box_id, x_cm, y_cm, rotation, side_name, size_name):
    """Formats the object data into the standard protocol string."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f'"{timestamp} {x_cm:.2f} {y_cm:.2f} {rotation:.1f} {side_name} {size_name}"'


def get_all_boxes_log_string(box_data):
    """Generates a list of log strings for all currently detected boxes."""
    log_strings = []
    for box_id, data in box_data.items():
        if data['x_cm'] is not None:
            log_strings.append(format_log_string(
                box_id, data['x_cm'], data['y_cm'], data['rotation'],
                data['side'], data['size']
            ))
    return log_strings


def load_calibration_config(config_path="calibration_config.json"):
    """
    Loads camera calibration data (focal length) and real marker size.
    Required to convert pixels into centimeters.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        settings = config.get("calibration_settings", {})
        output_params = config.get("output_parameters", {})
        real_marker_size_mm = settings.get("real_marker_size_mm")
        f_pixels = output_params.get("f_pixels")
        if not real_marker_size_mm or not f_pixels: return None
        return {'real_marker_size_mm': real_marker_size_mm, 'f_pixels': f_pixels}
    except:
        return None


def calculate_marker_position_2d(real_marker_size_mm, focal_length_pixels, marker_corners, frame_shape):
    """
    Mathematical Core: Calculates real-world coordinates (X, Y) and Rotation.
    Uses the pinhole camera model logic.

    1. Y (Distance) is calculated based on the perceived size of the marker vs real size.
    2. X (Lateral offset) is calculated based on the marker's deviation from the frame center.
    3. Rotation is calculated via the arctangent of the top edge vector.
    """
    # Calculate marker dimensions in pixels
    top_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
    bottom_width = np.linalg.norm(marker_corners[3] - marker_corners[2])
    left_height = np.linalg.norm(marker_corners[0] - marker_corners[3])
    right_height = np.linalg.norm(marker_corners[1] - marker_corners[2])
    
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    image_side = (avg_width + avg_height) / 2  # Average side length in pixels
    
    if image_side < 1.0: return None, None, None  # Avoid division by zero

    # Calculate Distance (Y) in mm: (RealSize * FocalLength) / ImageSize
    y_mm = (real_marker_size_mm * focal_length_pixels) / image_side
    
    # Calculate Lateral Offset (X)
    center_x_px = marker_corners[:, 0].mean()
    dx_px = center_x_px - (frame_shape[1] / 2)  # Deviation from center
    x_mm = (dx_px * y_mm) / focal_length_pixels

    # Calculate Rotation Angle
    vec_top = marker_corners[1] - marker_corners[0]
    rotation = math.degrees(math.atan2(vec_top[1], vec_top[0]))
    
    return x_mm, y_mm, rotation


def calculate_box_bbox(markers, padding=50):
    """Calculates a unified bounding box for a group of markers."""
    if not markers: return None
    all_points = np.array([c for m in markers for c in m['corners']])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    return (int(max(0, x_min - padding)), int(max(0, y_min - padding)),
            int(x_max + padding), int(y_max + padding))


def determine_box_side(marker_ids, box_id):
    """
    Determines which side of the cube is facing the camera based on visible Marker IDs.
    Each side has a specific range of IDs allocated.
    """
    if not marker_ids: return "Unknown"
    # Base ID depends on the Box Number (1-18, 19-36, 37-54)
    base_id = 1 if box_id == 1 else (19 if box_id == 2 else 37)
    detected_sides = []
    for mid in marker_ids:
        rel = mid - base_id
        if 0 <= rel <= 17: detected_sides.append(rel // 3)  # 3 markers per side
    if not detected_sides: return "Unknown"
    
    # Choose the side with the most visible markers
    most_common = max(set(detected_sides), key=detected_sides.count)
    return SIDE_NAMES[most_common] if 0 <= most_common < len(SIDE_NAMES) else "Unknown"


def generate_aruco_markers(aruco_dict_type, output_dir="markers"):
    """Generates ArUco marker images for printing if they don't exist."""
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.listdir(output_dir):
        d = aruco.getPredefinedDictionary(aruco_dict_type)
        for i in range(1, 55):
            cv2.imwrite(os.path.join(output_dir, f"aruco_{i}.png"),
                        aruco.generateImageMarker(d, i, 200))


class ArucoTag:
    """Wrapper class for ArUco detector initialization and detection."""
    def __init__(self, aruco_dict_type=aruco.DICT_6X6_250):
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
        self.parameters = aruco.DetectorParameters()
        # Subpixel refinement improves corner accuracy
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.minMarkerPerimeterRate = 0.02
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def detect(self, frame):
        """Returns detected markers, corners, and IDs."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        detected_info = []
        if ids is not None:
            for corner, mid in zip(corners, ids):
                mid = mid[0]
                if 1 <= mid <= 54:
                    pts = corner.reshape((4, 2))
                    cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                    detected_info.append({'id': mid, 'center': (cx, cy), 'corners': pts})
        return detected_info, corners, ids


def main():
    global server_running

    # 1. Setup Server
    ask_for_settings()

    # Start server in a Daemon thread (closes automatically when main program ends)
    t = threading.Thread(target=server_thread_func, daemon=True)
    t.start()

    # 2. Initialization
    initialize_log_file()
    generate_aruco_markers(aruco.DICT_6X6_250)
    calib = load_calibration_config()

    if not calib:
        print("Error: 'calibration_config.json' not found or invalid.")
        server_running = False
        return

    REAL_SZ, FOCAL = calib['real_marker_size_mm'], calib['f_pixels']
    detector = ArucoTag()

    # Initialize Camera (CAP_DSHOW is used on Windows for faster startup)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Camera not found")
        server_running = False
        return

    print("Camera started. Waiting for client in background...")
    print("Press 'q' to exit.")

    # History for coordinate smoothing
    history = {}
    fps_hist = deque(maxlen=30)
    prev_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success: break

            # Calculate FPS
            curr_time = time.time()
            fps_hist.append(1 / (curr_time - prev_time + 1e-6))
            prev_time = curr_time
            avg_fps = sum(fps_hist) / len(fps_hist)

            # 3. Detect Markers
            detections, corners, ids = detector.detect(frame)

            # 4. Group markers by Box ID (1, 2, 3)
            groups = {1: [], 2: [], 3: []}
            for det in detections:
                mid = det['id']
                if 1 <= mid <= 18:
                    groups[1].append(det)
                elif 19 <= mid <= 36:
                    groups[2].append(det)
                elif 37 <= mid <= 54:
                    groups[3].append(det)

            box_data = {}

            # 5. Process each Box
            for box_id, markers in groups.items():
                if not markers:
                    if box_id in history: del history[box_id]
                    continue

                # Initialize history for new box
                if box_id not in history:
                    history[box_id] = {
                        'x': deque(maxlen=SMOOTHING_WINDOW),
                        'y': deque(maxlen=SMOOTHING_WINDOW),
                        'rot': deque(maxlen=SMOOTHING_WINDOW)
                    }

                raw_x, raw_y, raw_rot = [], [], []
                px_cx, px_cy = [], []
                m_ids = []

                # Calculate position for each marker individually
                for m in markers:
                    m_ids.append(m['id'])
                    px_cx.append(m['center'][0]);
                    px_cy.append(m['center'][1])
                    x, y, r = calculate_marker_position_2d(REAL_SZ, FOCAL, m['corners'], frame.shape)
                    if x is not None:
                        raw_x.append(x);
                        raw_y.append(y);
                        raw_rot.append(r)

                # Average and smooth the data
                if raw_x:
                    curr_x = np.mean(raw_x) / 10.0  # Convert mm to cm
                    curr_y = np.mean(raw_y) / 10.0
                    curr_r = np.mean(raw_rot)

                    history[box_id]['x'].append(curr_x)
                    history[box_id]['y'].append(curr_y)
                    history[box_id]['rot'].append(curr_r)

                    smooth_x = sum(history[box_id]['x']) / len(history[box_id]['x'])
                    smooth_y = sum(history[box_id]['y']) / len(history[box_id]['y'])
                    smooth_r = sum(history[box_id]['rot']) / len(history[box_id]['rot'])

                    bbox = calculate_box_bbox(markers)
                    side = determine_box_side(m_ids, box_id)
                    size = BOX_SIZES.get(box_id, "Unknown")
                    centroid = (int(np.mean(px_cx)), int(np.mean(px_cy)))

                    box_data[box_id] = {
                        'x_cm': smooth_x,
                        'y_cm': smooth_y,
                        'rotation': smooth_r,
                        'count': len(markers),
                        'bbox': bbox,
                        'side': side,
                        'size': size,
                        'centroid': centroid
                    }

            # 6. Visualization (Draw rectangles and text)
            annotated = frame.copy()

            for bid, d in box_data.items():
                x1, y1, x2, y2 = d['bbox']
                # Color depends on Box ID
                col = (255, 0, 0) if bid == 1 else ((0, 255, 0) if bid == 2 else (0, 0, 255))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
                label = f"[{d['size']}] {d['side']} {int(d['y_cm'])}cm"

                # Place label near centroid or top-left
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                if d['count'] >= 2:
                    tx, ty = d['centroid'][0] - w // 2, d['centroid'][1] + h // 2
                else:
                    tx, ty = x1 + 5, y1 - 10

                cv2.rectangle(annotated, (tx - 5, ty - h - 5), (tx + w + 5, ty + 5), col, -1)
                cv2.putText(annotated, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if ids is not None: aruco.drawDetectedMarkers(annotated, corners, ids)

            # Draw Connection Status and FPS
            status_color = (0, 255, 0) if active_client_conn else (0, 0, 255)
            status_text = "CLIENT: CONNECTED" if active_client_conn else "CLIENT: WAITING..."
            cv2.putText(annotated, status_text, (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_off = 60
            for bid, d in box_data.items():
                txt = f"Box{bid}: X:{d['x_cm']:.1f} Y:{d['y_cm']:.0f} R:{d['rotation']:.0f}"
                cv2.putText(annotated, txt, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_off += 25

            cv2.imshow("ArUco Server Tracker", annotated)

            # 7. Keyboard Handling
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):  # Quit
                break
            elif k == ord('c'):  # Clear in-memory log
                clear_log_string()
            elif 65 <= k <= 90 or 97 <= k <= 122:  # Any letter triggers logging/sending
                ls = get_all_boxes_log_string(box_data)

                if ls:
                    for s in ls:
                        write_to_log(s)        # Write to file
                        send_data_to_client(s) # Send via TCP

                    if active_client_conn:
                        print(f"[SENT] Sent {len(ls)} lines to client.")
                    else:
                        print(f"[LOGGED] Saved {len(ls)} lines (Client not connected).")

    finally:
        # Safe exit cleanup
        server_running = False
        if active_client_conn:
            active_client_conn.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
