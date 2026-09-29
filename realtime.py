import cv2
import numpy as np
import joblib
import time
from imutils import face_utils

from feature_extraction import (
    eye_aspect_ratio, mouth_aspect_ratio, head_tilt_angle, head_nod_ratio,
    detector, predictor, lStart, lEnd, rStart, rEnd, mStart, mEnd
)
from preprocessing import preprocess_frame

EAR_THRESHOLD      = 0.20
MAR_THRESHOLD      = 0.60
EAR_CONSEC_FRAMES  = 20
YAWN_CONSEC_FRAMES = 15

model  = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")


def play_alert():
    try:
        import winsound
        winsound.Beep(1000, 500)
    except:
        print("\a")


def run():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    eye_closed_counter = 0
    yawn_counter       = 0
    blink_count        = 0
    eye_was_closed     = False

    print("Running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame           = cv2.flip(frame, 1)
        gray, equalized = preprocess_frame(frame)
        faces           = detector(equalized, 0)

        status       = "ALERT"
        status_color = (0, 200, 0)

        for face in faces:
            shape    = predictor(equalized, face)
            shape    = face_utils.shape_to_np(shape)
            leftEye  = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth    = shape[mStart:mEnd]

            left_ear  = eye_aspect_ratio(leftEye)
            right_ear = eye_aspect_ratio(rightEye)
            avg_ear   = (left_ear + right_ear) / 2.0
            mar       = mouth_aspect_ratio(mouth)
            tilt      = head_tilt_angle(shape)
            nod       = head_nod_ratio(shape)

            if avg_ear < EAR_THRESHOLD:
                eye_closed_counter += 1
                eye_was_closed      = True
            else:
                if eye_was_closed and eye_closed_counter >= 2:
                    blink_count += 1
                eye_was_closed     = False
                eye_closed_counter = 0

            yawn_counter = yawn_counter + 1 if mar > MAR_THRESHOLD else 0

            fv   = scaler.transform([[avg_ear, mar, tilt, nod, left_ear, right_ear]])
            pred = model.predict(fv)[0]
            prob = model.predict_proba(fv)[0][1]

            is_drowsy = (
                pred == 1 or
                eye_closed_counter >= EAR_CONSEC_FRAMES or
                yawn_counter >= YAWN_CONSEC_FRAMES
            )

            if is_drowsy:
                status       = "DROWSY!"
                status_color = (0, 0, 255)
                play_alert()

            color = (0, 0, 255) if is_drowsy else (0, 200, 0)
            cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, color, 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, color, 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)],    -1, (255, 220, 0), 1)

            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            ear_col = (0, 0, 255) if avg_ear < EAR_THRESHOLD else (0, 255, 0)
            mar_col = (0, 0, 255) if mar > MAR_THRESHOLD     else (0, 255, 0)

            cv2.putText(frame, f"EAR:    {avg_ear:.2f}", (10,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_col, 2)
            cv2.putText(frame, f"MAR:    {mar:.2f}",     (10,  60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mar_col, 2)
            cv2.putText(frame, f"Tilt:   {tilt:.1f}",   (10,  90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
            cv2.putText(frame, f"Drowsy: {prob:.0%}",   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 0), 2)

        cv2.rectangle(frame, (0, 0), (640, 38), (20, 20, 20), -1)
        cv2.putText(frame, f"STATUS: {status}", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()