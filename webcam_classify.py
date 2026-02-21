import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam classification with your custom classes")
    parser.add_argument(
        "--model",
        default="runs/classify/train/weights/best.pt",
        help="Path to YOLO classification model",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--imgsz", type=int, default=224, help="Inference size")
    parser.add_argument("--device", default="cpu", help="cpu or cuda device id")
    parser.add_argument("--topk", type=int, default=1, help="How many top classes to show")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")

    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            source=frame,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        probs = result.probs
        if probs is not None:
            top_indices = probs.top5[: args.topk]
            lines = []
            for idx in top_indices:
                idx = int(idx)
                name = result.names[idx]
                conf = float(probs.data[idx])
                lines.append(f"{name}: {conf:.2f}")

            y = 30
            for line in lines:
                cv2.putText(
                    frame,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                y += 32

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (12, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Full-frame box to mimic overlay style; this is classification, not object localization.
        cv2.rectangle(frame, (2, 2), (frame.shape[1] - 2, frame.shape[0] - 2), (0, 255, 0), 2)
        cv2.imshow("Garbage Classifier (Q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
