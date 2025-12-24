import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
from collections import deque

CONFIG_FILE = "../calibration_config.json"
HISTORY_LENGTH = 20

DEFAULT_CONFIG = {
    "calibration_settings": {
        "real_marker_size_mm": 50.0,
        "known_distance_mm": 300.0,
        "calibration_marker_id": 1
    },
    "output_parameters": {
        "f_pixels": None,
        "aruco_dict_type": "DICT_6X6_250"
    }
}


def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"Конфігураційний файл '{CONFIG_FILE}' не знайдено. Створюю новий з поясненнями...")
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                config_with_comments = {
                    "INFO": "Будь ласка, ЗАДАЙТЕ ваші РЕАЛЬНІ виміри в секції 'calibration_settings' перед калібруванням.",
                    **DEFAULT_CONFIG
                }
                json.dump(config_with_comments, f, indent=4, ensure_ascii=False)
            print(f"Файл '{CONFIG_FILE}' створено. ОБОВ'ЯЗКОВО ВНЕСІТЬ СВОЇ ВИМІРИ!")
            return DEFAULT_CONFIG['calibration_settings'], DEFAULT_CONFIG['output_parameters'], True
        except Exception as e:
            print(f"ПОМИЛКА при створенні JSON: {e}")
            return None, None, True

    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
            settings = full_config.get('calibration_settings', DEFAULT_CONFIG['calibration_settings'])
            params = full_config.get('output_parameters', DEFAULT_CONFIG['output_parameters'])
            return settings, params, False
    except Exception as e:
        print(f"ПОМИЛКА читання/парсингу JSON: {e}. Використовуються типові значення.")
        return DEFAULT_CONFIG['calibration_settings'], DEFAULT_CONFIG['output_parameters'], False


def save_focus_result(F_pixels):
    try:
        with open(CONFIG_FILE, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            data['output_parameters']['f_pixels'] = round(F_pixels, 2)

            f.seek(0)
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.truncate()
        print(f"\nУСЕРЕДНЕНЕ значення F_pixels: {F_pixels:.2f} px успішно збережено у '{CONFIG_FILE}'.")
    except Exception as e:
        print(f"ПОМИЛКА при збереженні результату в JSON: {e}")

def detect_aruco_and_measure(frame, calibration_marker_id, aruco_dict_type):
    try:
        aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, aruco_dict_type))
    except AttributeError:
        return None, None

    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    pixel_width = None

    if ids is not None and len(ids) > 0:
        for corner, mid in zip(corners, ids):
            mid = mid[0]
            if mid == calibration_marker_id:
                pts = corner.reshape((4, 2))
                width_top = np.linalg.norm(pts[0] - pts[1])
                width_bottom = np.linalg.norm(pts[3] - pts[2])
                pixel_width = (width_top + width_bottom) / 2

                cv2.polylines(frame, [np.int32(pts)], True, (0, 255, 0), 2)
                center_x = int(pts[:, 0].mean())
                center_y = int(pts[:, 1].mean())

                return pixel_width, (center_x, center_y)

    return None, None

def main():
    settings, params, just_created = load_config()

    if settings is None:
        return

    if just_created:
        print("\n--- УВАГА! ПЕРЕД ЗАПУСКОМ ---")
        print(f"Змініть значення 'real_marker_size_mm' та 'known_distance_mm' у файлі '{CONFIG_FILE}'")
        print("та перезапустіть скрипт!")
        return

    REAL_MARKER_SIZE_MM = settings['real_marker_size_mm']
    KNOWN_DISTANCE_MM = settings['known_distance_mm']
    CALIBRATION_MARKER_ID = settings['calibration_marker_id']
    ARUCO_DICT_TYPE = params['aruco_dict_type']

    f_pixels_history = deque(maxlen=HISTORY_LENGTH)

    print("--- 🔬 Режим Калібрування Фокусної Відстані ---")
    print(f"   Формула: F_pixels = (P_img * D_known) / S_real")
    print(f"   [КОРИСТУВАЧ]: S_real ({REAL_MARKER_SIZE_MM} мм), D_known ({KNOWN_DISTANCE_MM} мм)")
    print(f"   [СКРИПТ]: Вимірює P_img (пікселі) та обчислює F_pixels.")
    print("   Натисніть 'c', щоб зберегти УСЕРЕДНЕНЕ значення F_pixels.")
    print("====================================================================")

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ПОМИЛКА: Не вдалося відкрити камеру.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pixel_width, center = detect_aruco_and_measure(frame, CALIBRATION_MARKER_ID, ARUCO_DICT_TYPE)

        current_f_pixels = None
        current_status = ""

        if pixel_width is not None and pixel_width > 0:
            current_f_pixels = (pixel_width * KNOWN_DISTANCE_MM) / REAL_MARKER_SIZE_MM

            f_pixels_history.append(current_f_pixels)
            avg_f_pixels = np.mean(f_pixels_history)

            current_status = f"F_CUR: {current_f_pixels:.1f} px | F_AVG ({len(f_pixels_history)}/{HISTORY_LENGTH}): {avg_f_pixels:.1f} px"

            cv2.putText(frame, f"P_img: {pixel_width:.1f} px", (center[0] - 50, center[1] + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            current_status = f"Маркер ID {CALIBRATION_MARKER_ID} не знайдено."

        cv2.putText(frame, current_status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Focus Calibration (REAL-TIME AVG)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(f_pixels_history) > 0:
                final_f_pixels = np.mean(f_pixels_history)
                save_focus_result(final_f_pixels)
                break
            else:
                print("ПОМИЛКА: Маркер жодного разу не був знайдений. Неможливо виконати калібрування.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Калібрування завершено.")


if __name__ == '__main__':
    main()
