import cv2
import numpy as np
import os


def preprocess_frame(frame):
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized   = cv2.resize(gray, (640, 480))
    blurred   = cv2.GaussianBlur(resized, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    return gray, equalized


def preprocess_image_file(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    gray, equalized = preprocess_frame(img)
    return img, equalized


def extract_frames_from_video(video_path, output_folder, frame_interval=5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            out_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {frame_count} total frames")
    return saved_count


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray, equalized = preprocess_frame(frame)
        cv2.imshow("Original",     frame)
        cv2.imshow("Preprocessed", equalized)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()