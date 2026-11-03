"""Optional webcam demo; requires requirements-vision.txt."""
def run_camera(model_path: str = "yolo11n.pt") -> None:
    import cv2
    from ultralytics import YOLO
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera unavailable; check macOS permissions")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model(frame, verbose=False)
            cv2.imshow("NOVA - Detection Only", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
