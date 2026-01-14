import os, time, json, threading, cv2
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from detector import process_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "detection_log.json")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

latest_frame = None
frame_lock = threading.Lock()

LOG_COOLDOWN = 2
_last_log_time = 0


def save_log(status, confidence):
    global _last_log_time
    now = time.time()

    if now - _last_log_time < LOG_COOLDOWN:
        return
    _last_log_time = now

    log = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "confidence": confidence
    }

    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)

    logs.insert(0, log)
    logs = logs[:50]

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)


@app.route("/upload", methods=["POST"])
def upload():
    global latest_frame

    data = request.get_data()
    if not data:
        return jsonify({"error": "no image"}), 400

    raw_path = os.path.join(UPLOAD_DIR, f"{int(time.time())}.jpg")
    with open(raw_path, "wb") as f:
        f.write(data)

    out_path = os.path.join(STATIC_DIR, "latest.jpg")
    out_path, info = process_image(raw_path, out_path)

    frame = cv2.imread(out_path)
    with frame_lock:
        latest_frame = frame

    if info["covered"]:
        save_log("COVERED", info["confidence"])
    elif info["person"]:
        save_log("PERSON", info["confidence"])
    else:
        save_log("NONE", 0)

    return jsonify(info), 200


@app.route("/stream")
def stream():
    def gen():
        while True:
            if os.path.exists(os.path.join(STATIC_DIR, "latest.jpg")):
                with open(os.path.join(STATIC_DIR, "latest.jpg"), "rb") as f:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + f.read()
                        + b"\r\n"
                    )
            time.sleep(0.05)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/logs")
def logs():
    if not os.path.exists(LOG_FILE):
        return jsonify([])
    with open(LOG_FILE, "r") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
