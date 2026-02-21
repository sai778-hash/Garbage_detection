import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time webcam detection with custom garbage classification on each box"
    )
    parser.add_argument(
        "--det-model",
        default="yolov8n.pt",
        help="Detection model used only for locating boxes",
    )
    parser.add_argument(
        "--cls-model",
        default="runs/classify/train/weights/best.pt",
        help="Classification model trained on your garbage classes",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--det-conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--cls-conf", type=float, default=0.40, help="Classification confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Detection image size")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="Classification image size")
    parser.add_argument("--device", default="cpu", help="cpu, 0, 0,1 etc.")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    det_model = YOLO(args.det_model)
    cls_model = YOLO(args.cls_model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. Try --camera 1 or check camera permissions."
        )

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        det_result = det_model.predict(
            source=frame,
            conf=args.det_conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        annotated = frame.copy()
        boxes = det_result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cls_result = cls_model.predict(
                    source=crop,
                    imgsz=args.cls_imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                probs = cls_result.probs
                if probs is None:
                    continue

                top1_index = int(probs.top1)
                top1_conf = float(probs.top1conf)
                if top1_conf < args.cls_conf:
                    continue

                class_name = cls_result.names[top1_index]
                label = f"{class_name} {top1_conf:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), args.line_width)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Garbage Detection - Press Q to exit", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
