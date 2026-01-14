import os
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "best.pt"
)

model = YOLO(MODEL_PATH)


def process_image(input_path, output_path):
    img = cv2.imread(input_path)

    info = {
        "person": False,
        "covered": False,
        "confidence": 0.0
    }

    results = model(img, conf=0.2)[0]

    best_conf = 0.0
    best_status = None

    for box in results.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = (0, 255, 0)

        if name == "covered_faces":
            color = (0, 0, 255)
            info["covered"] = True

            if conf > best_conf:
                best_conf = conf
                best_status = "covered"

        elif name == "normal_faces":
            info["person"] = True

            if conf > best_conf and best_status != "covered":
                best_conf = conf
                best_status = "person"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{name} {conf:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    info["confidence"] = round(best_conf, 2)
    cv2.imwrite(output_path, img)

    return output_path, info
