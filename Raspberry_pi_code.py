import os, time, threading, cv2, numpy as np
from collections import deque
from datetime import datetime
from flask import Flask, Response, make_response
from ultralytics import YOLO
import RPi.GPIO as GPIO

# ============================================================
# -------------------- CAMERA / YOLO CONFIG ------------------
# ============================================================
# Phone IP camera URL (prefer HTTP + IPv4)
# Example: "http://192.168.43.1:8080/video"
CAM_INDEX = "http://10.16.180.56:8080/video"   # <-- CHANGE IF NEEDED

FRAME_W, FRAME_H = 640, 480
TARGET_FPS = 30

# --- Models (EDIT THESE PATHS) ---
PERSON_MODEL_PATH      = os.path.expanduser("/home/raspberry-pi/ai/person.pt")
PERSON_MODEL_FALLBACK  = "yolov8n.pt"
PLATE_MODEL_PATH       = os.path.expanduser("/home/raspberry-pi/ai/plate.pt")

CONF_PERSON     = 0.35
CONF_PLATE      = 0.20      # slightly lower to make plates easier to detect
DETECT_IMGSZ    = 640       # higher size helps small plates
RUN_FACE_EVERY_N  = 2
RUN_PLATE_EVERY_N = 2

USE_FACE = True

RUN_OCR   = True
OCR_PSM   = 7
OCR_PADPX = 2
try:
    import pytesseract
    HAS_TESS = True
except Exception:
    HAS_TESS = False

SAVE_GAP_SEC = 5
CAPTURE_DIR  = os.path.expanduser('~/survobot/captures')
PLATE_DIR    = os.path.expanduser('~/survobot/plates')

MJPEG_QUALITY = 65
STREAM_SLEEP  = 0.005

# ============================================================
# ----------------- ROBOT / ULTRASONIC CONFIG ----------------
# ============================================================
USE_ULTRASONIC = True   # True = robot moves with obstacle avoidance

# Use BOARD numbering (physical pins)
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# ----- MOTOR PINS (L298N) -----
# Left motor (Motor A)
M1_EN  = 12   # ENA
M1_IN1 = 16   # IN1
M1_IN2 = 38   # IN2

# Right motor (Motor B)
M2_EN  = 35   # ENB
M2_IN1 = 13   # IN3
M2_IN2 = 37   # IN4

for pin in [M1_EN, M1_IN1, M1_IN2, M2_EN, M2_IN1, M2_IN2]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# ----- ULTRASONIC PINS (3 SENSORS) -----
TRIG_FRONT = 33
ECHO_FRONT = 29

TRIG_LEFT  = 32
ECHO_LEFT  = 22

TRIG_RIGHT = 18
ECHO_RIGHT = 10

GPIO.setup(TRIG_FRONT, GPIO.OUT)
GPIO.setup(TRIG_LEFT,  GPIO.OUT)
GPIO.setup(TRIG_RIGHT, GPIO.OUT)

# Use pull-down so echo pins don't float
GPIO.setup(ECHO_FRONT, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(ECHO_LEFT,  GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(ECHO_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Distances / timings
STOP_DISTANCE_CM = 20.0   # act only if front < 20 cm (as in your log)
TURN_DISTANCE_CM = 25.0
TURN_TIME        = 0.4
BACK_TIME        = 0.4

# Sound / limits
SOUND_SPEED = 34300.0  # cm/s
MIN_DIST_CM = 2.0
MAX_DIST_CM = 400.0

# ============================================================
# -------------------- DIRECTORIES & MODELS ------------------
# ============================================================
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(PLATE_DIR, exist_ok=True)
try:
    cv2.setNumThreads(1)
except:
    pass

# Person model
if os.path.isfile(PERSON_MODEL_PATH):
    person_model_path = PERSON_MODEL_PATH
else:
    person_model_path = PERSON_MODEL_FALLBACK
    print(f"[WARN] Using fallback person model: {PERSON_MODEL_FALLBACK}")
person_model = YOLO(person_model_path)
person_class_ids = None

# Plate model
USE_PLATE_MODEL = os.path.isfile(PLATE_MODEL_PATH)
lp_model = YOLO(PLATE_MODEL_PATH) if USE_PLATE_MODEL else None
if USE_PLATE_MODEL:
    print(f"[INFO] Plate model: {PLATE_MODEL_PATH}")
else:
    print(f"[WARN] Plate model missing: {PLATE_MODEL_PATH} (plate detection disabled)")

# Face cascade
if USE_FACE:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if face_cascade.empty():
        print("[WARN] Face cascade could not be loaded. Disabling face detection.")
        USE_FACE = False

# ============================================================
# -------------------- IP CAMERA READER ----------------------
# ============================================================
class IPcamReader:
    """
    Low-latency IP camera reader:
    - Background thread continuously reads frames
    - Stores only the latest frame (drops old ones)
    """
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print("[WARN] FFMPEG backend failed, trying default backend...")
            self.cap = cv2.VideoCapture(self.url)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open stream: {self.url}")

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("[INFO] IPcamReader: capture thread started")

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                print("[WARN] IPcamReader: failed to grab frame, retrying...")
                time.sleep(0.1)
                continue

            # Optional downscale if needed:
            # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

            with self.lock:
                self.latest_frame = frame

        print("[INFO] IPcamReader: capture thread stopped")

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        try:
            self.cap.release()
        except Exception:
            pass

reader = None

# ============================================================
# ---------------------- ROBOT FUNCTIONS ---------------------
# ============================================================
def left():
    GPIO.output(M1_EN, GPIO.HIGH)
    GPIO.output(M2_EN, GPIO.HIGH)

    GPIO.output(M1_IN1, GPIO.HIGH)
    GPIO.output(M1_IN2, GPIO.LOW)

    GPIO.output(M2_IN1, GPIO.HIGH)
    GPIO.output(M2_IN2, GPIO.LOW)


def right():
    GPIO.output(M1_EN, GPIO.HIGH)
    GPIO.output(M2_EN, GPIO.HIGH)

    GPIO.output(M1_IN1, GPIO.LOW)
    GPIO.output(M1_IN2, GPIO.HIGH)

    GPIO.output(M2_IN1, GPIO.LOW)
    GPIO.output(M2_IN2, GPIO.HIGH)


def forward():
    GPIO.output(M1_EN, GPIO.HIGH)
    GPIO.output(M2_EN, GPIO.HIGH)

    # left motor backward
    GPIO.output(M1_IN1, GPIO.LOW)
    GPIO.output(M1_IN2, GPIO.HIGH)

    # right motor forward
    GPIO.output(M2_IN1, GPIO.HIGH)
    GPIO.output(M2_IN2, GPIO.LOW)


def backward():
    GPIO.output(M1_EN, GPIO.HIGH)
    GPIO.output(M2_EN, GPIO.HIGH)

    # left motor forward
    GPIO.output(M1_IN1, GPIO.HIGH)
    GPIO.output(M1_IN2, GPIO.LOW)

    # right motor backward
    GPIO.output(M2_IN1, GPIO.LOW)
    GPIO.output(M2_IN2, GPIO.HIGH)


def stop_motors():
    GPIO.output(M1_IN1, GPIO.LOW)
    GPIO.output(M1_IN2, GPIO.LOW)
    GPIO.output(M2_IN1, GPIO.LOW)
    GPIO.output(M2_IN2, GPIO.LOW)
    GPIO.output(M1_EN, GPIO.LOW)
    GPIO.output(M2_EN, GPIO.LOW)

# -------- Filtered distance functions --------
def measure_distance_raw(trig_pin, echo_pin, timeout=0.03):
    """Single raw measurement. Returns distance in cm or None if invalid."""
    GPIO.output(trig_pin, GPIO.LOW)
    time.sleep(0.0002)

    GPIO.output(trig_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig_pin, GPIO.LOW)

    t0 = time.time()

    # Wait for echo HIGH
    while GPIO.input(echo_pin) == GPIO.LOW:
        if time.time() - t0 > timeout:
            return None
    start = time.time()

    # Wait for echo LOW
    while GPIO.input(echo_pin) == GPIO.HIGH:
        if time.time() - t0 > timeout:
            return None
    end = time.time()

    elapsed = end - start
    dist_cm = (elapsed * SOUND_SPEED) / 2.0

    if dist_cm < MIN_DIST_CM or dist_cm > MAX_DIST_CM:
        return None
    return dist_cm


def get_distance_cm(trig_pin, echo_pin, samples=3, delay=0.01):
    """Median of multiple raw samples; if all fail, returns 999.0."""
    vals = []
    for _ in range(samples):
        d = measure_distance_raw(trig_pin, echo_pin)
        if d is not None:
            vals.append(d)
        time.sleep(delay)

    if not vals:
        return 999.0
    vals.sort()
    return vals[len(vals) // 2]

# ============================================================
# ----------------------- GLOBAL STATE -----------------------
# ============================================================
running = True
disp_q = deque(maxlen=1)

det_lock = threading.Lock()
last_det = {"persons": [], "faces": [], "plates": []}
last_save_ts = 0.0

# ============================================================
# ----------------------- OCR HELPERS ------------------------
# ============================================================
def ocr_text(crop_bgr, psm=7):
    if not RUN_OCR or not HAS_TESS or crop_bgr is None or crop_bgr.size == 0:
        return ""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (int(g.shape[1] * 2), int(g.shape[0] * 2)), interpolation=cv2.INTER_CUBIC)
    g = cv2.equalizeHist(g)
    g = cv2.bilateralFilter(g, 7, 35, 35)
    bw = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )
    bw_inv = 255 - bw
    cfg = (
        f'--oem 3 --psm {psm} -l eng '
        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    import pytesseract as _pt
    t1 = _pt.image_to_string(bw, config=cfg)
    t2 = _pt.image_to_string(bw_inv, config=cfg)
    clean = lambda s: ''.join(ch for ch in s if ch.isalnum())
    c1, c2 = clean(t1), clean(t2)
    return c1 if len(c1) >= len(c2) else c2

def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W - 1))
    y2 = max(0, min(int(y2), H - 1))
    return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None

# ============================================================
# ---------------------- DETECT HELPERS ----------------------
# ============================================================
def resolve_person_boxes(results, shape):
    boxes = []
    if not results:
        return boxes
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return boxes
    global person_class_ids
    if person_class_ids is None:
        try:
            names = r.names if hasattr(r, 'names') else person_model.names
            if names and isinstance(names, dict):
                person_class_ids = [
                    i for i, n in names.items()
                    if str(n).lower() == 'person'
                ]
                if not person_class_ids and len(names) == 1:
                    person_class_ids = list(names.keys())
            else:
                person_class_ids = [0]
        except Exception:
            person_class_ids = [0]
    H, W = shape[:2]
    for b in r.boxes:
        cls = int(b.cls[0].item()) if b.cls is not None else 0
        if cls in person_class_ids:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H - 1))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return boxes

# ============================================================
# ------------------------ THREADS ---------------------------
# ============================================================
def detect_loop():
    global reader
    frame_id = 0
    while running:
        if reader is None:
            time.sleep(0.01)
            continue
        frame = reader.read()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_id += 1
        H, W = frame.shape[:2]

        scale = DETECT_IMGSZ / max(H, W)
        if scale < 1.0:
            det_img = cv2.resize(
                frame,
                (int(W * scale), int(H * scale)),
                interpolation=cv2.INTER_AREA
            )
        else:
            det_img = frame
        inv = (1.0 / scale) if scale < 1.0 else 1.0

        # Persons
        r = person_model.predict(
            source=det_img,
            imgsz=DETECT_IMGSZ,
            conf=CONF_PERSON,
            verbose=False
        )
        small = resolve_person_boxes(r, det_img.shape)
        persons = [
            [int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv)]
            for x1, y1, x2, y2 in small
        ]

        # Faces
        faces = []
        if USE_FACE and (frame_id % RUN_FACE_EVERY_N == 0):
            gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
            fsmall = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )
            # Debug: number of faces found
            # print(f"[DEBUG] frame {frame_id}: faces={len(fsmall)}")
            for (x, y, w, h) in fsmall:
                faces.append(
                    (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
                )

        # Plates + OCR (draw even if OCR fails)
        plates = []
        if USE_PLATE_MODEL and (frame_id % RUN_PLATE_EVERY_N == 0):
            lp_res = lp_model.predict(
                source=det_img,
                imgsz=DETECT_IMGSZ,
                conf=CONF_PLATE,
                device='cpu',
                verbose=False
            )
            if lp_res and lp_res[0].boxes is not None and len(lp_res[0].boxes) > 0:
                print(f"[DEBUG] frame {frame_id}: plate boxes detected = {len(lp_res[0].boxes)}")
                for b in lp_res[0].boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                    x1 = int(x1 * inv) - OCR_PADPX
                    y1 = int(y1 * inv) - OCR_PADPX
                    x2 = int(x2 * inv) + OCR_PADPX
                    y2 = int(y2 * inv) + OCR_PADPX
                    clamped = clamp_box(x1, y1, x2, y2, W, H)
                    if clamped is None:
                        continue
                    x1, y1, x2, y2 = clamped
                    crop = frame[y1:y2, x1:x2].copy()

                    text = ""
                    if RUN_OCR and HAS_TESS:
                        text = ocr_text(crop, psm=OCR_PSM)

                    label = text if text else "PLATE"
                    plates.append((label, (x1, y1, x2, y2)))

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    fname_label = text if text else "plate"
                    cv2.imwrite(
                        os.path.join(PLATE_DIR, f"{ts}_{fname_label}_crop.jpg"),
                        crop
                    )

        with det_lock:
            last_det["persons"] = persons
            last_det["faces"]   = faces
            last_det["plates"]  = plates

def compose_loop():
    global last_save_ts, reader
    fps_t0, fps_n, fps_val = time.time(), 0, 0.0
    while running:
        if reader is None:
            time.sleep(0.01)
            continue
        frame = reader.read()
        if frame is None:
            time.sleep(0.003)
            continue

        frame = frame.copy()
        H, W = frame.shape[:2]

        with det_lock:
            persons = list(last_det["persons"])
            faces   = list(last_det["faces"])
            plates  = list(last_det["plates"])

        # Bigger font & thickness
        label_font      = cv2.FONT_HERSHEY_SIMPLEX
        label_person_fs = 0.9
        label_face_fs   = 0.9
        label_plate_fs  = 0.9
        label_thickness = 2

        for (x1, y1, x2, y2) in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, "person",
                (x1, max(20, y1 - 8)),
                label_font, label_person_fs, (0, 255, 0), label_thickness
            )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            cv2.putText(
                frame, "face",
                (x, max(20, y - 8)),
                label_font, label_face_fs, (255, 200, 0), label_thickness
            )
        for (txt, (x1, y1, x2, y2)) in plates:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(
                frame, f"PLATE:{txt}",
                (x1, max(25, y1 - 10)),
                label_font, label_plate_fs, (0, 140, 255), label_thickness
            )

        if persons and (time.time() - last_save_ts) > SAVE_GAP_SEC:
            fname = datetime.now().strftime("%Y%m%d_%H%M%S_person.jpg")
            cv2.imwrite(os.path.join(CAPTURE_DIR, fname), frame)
            last_save_ts = time.time()

        fps_n += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            fps_val = fps_n / (now - fps_t0)
            fps_n, fps_t0 = 0, now
        cv2.putText(
            frame, f"{fps_val:4.1f} FPS",
            (W - 140, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
        )

        disp_q.append(frame)
        time.sleep(0.001)

def ultrasonic_motion_loop():
    if not USE_ULTRASONIC:
        return
    print(f"[INFO] Ultrasonic motion thread started. Front condition: act only if d_front < {STOP_DISTANCE_CM} cm")
    try:
        while running:
            d_front = get_distance_cm(TRIG_FRONT, ECHO_FRONT)
            print(f"front {d_front:.2f} cm", end="\r")

            if d_front < STOP_DISTANCE_CM:
                stop_motors()
                time.sleep(0.05)

                d_left  = get_distance_cm(TRIG_LEFT,  ECHO_LEFT)
                d_right = get_distance_cm(TRIG_RIGHT, ECHO_RIGHT)
                print(f"\nFront<{STOP_DISTANCE_CM}cm | L={d_left:.1f} R={d_right:.1f}")

                if d_left > d_right and d_left > TURN_DISTANCE_CM:
                    print("Turning LEFT")
                    left()
                    time.sleep(TURN_TIME)
                    stop_motors()
                elif d_right >= d_left and d_right > TURN_DISTANCE_CM:
                    print("Turning RIGHT")
                    right()
                    time.sleep(TURN_TIME)
                    stop_motors()
                else:
                    print("Blocked both sides, going BACKWARD")
                    backward()
                    time.sleep(BACK_TIME)
                    stop_motors()
            else:
                forward()

            time.sleep(0.05)
    finally:
        stop_motors()
        print("[INFO] Ultrasonic motion thread exiting")

# ============================================================
# ---------------------- FLASK STREAMING ---------------------
# ============================================================
app = Flask(__name__)

@app.after_request
def nocache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp

@app.route("/")
def index():
    html = """
    <!doctype html><title>SurvoBot — Live</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <style>
        body{margin:0;background:#111;color:#eee;font-family:system-ui}
        header{padding:10px}
        img{width:100vw;height:auto}
    </style>
    <header><h3>SurvoBot — Live</h3><p>Stream: /video_feed</p></header>
    <img src="/video_feed">
    """
    return make_response(html)

def mjpeg_gen():
    while True:
        if not disp_q:
            time.sleep(STREAM_SLEEP)
            continue
        frame = disp_q[-1]
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])
        if not ok:
            time.sleep(STREAM_SLEEP)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg.tobytes() +
            b"\r\n"
        )
        time.sleep(STREAM_SLEEP)

@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ============================================================
# -------------------- START THREADS SAFE --------------------
# ============================================================
_threads_started = False
def start_threads():
    global _threads_started, reader
    if _threads_started:
        return
    _threads_started = True

    if reader is None:
        reader = IPcamReader(CAM_INDEX)
        reader.start()

    threading.Thread(target=detect_loop,             daemon=True).start()
    threading.Thread(target=compose_loop,            daemon=True).start()
    threading.Thread(target=ultrasonic_motion_loop,  daemon=True).start()

start_threads()

# ============================================================
# --------------------------- MAIN ---------------------------
# ============================================================
def main():
    if RUN_OCR and not HAS_TESS:
        print("[WARN] RUN_OCR=True but Tesseract not found. Install 'tesseract-ocr' or set RUN_OCR=False.")
    start_threads()
    print("[INFO] Flask starting on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt, stopping…")
    finally:
        running = False
        time.sleep(0.2)
        try:
            if reader is not None:
                reader.stop()
        except Exception:
            pass
        try:
            stop_motors()
            GPIO.cleanup()
        except Exception:
            pass
        print("[INFO] Clean exit.")
