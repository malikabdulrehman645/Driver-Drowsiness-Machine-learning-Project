import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

from preprocessing import preprocess_frame

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

NOSE_TIP         = 30
CHIN             = 8
LEFT_EYE_CORNER  = 36
RIGHT_EYE_CORNER = 45


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2],  mouth[10])
    B = dist.euclidean(mouth[4],  mouth[8])
    C = dist.euclidean(mouth[0],  mouth[6])
    return (A + B) / (2.0 * C)


def head_tilt_angle(shape):
    left  = shape[LEFT_EYE_CORNER]
    right = shape[RIGHT_EYE_CORNER]
    dx    = right[0] - left[0]
    dy    = right[1] - left[1]
    return abs(np.degrees(np.arctan2(dy, dx)))


def head_nod_ratio(shape):
    nose        = shape[NOSE_TIP]
    chin        = shape[CHIN]
    left        = shape[LEFT_EYE_CORNER]
    right       = shape[RIGHT_EYE_CORNER]
    face_height = dist.euclidean(nose, chin)
    face_width  = dist.euclidean(left, right)
    if face_width == 0:
        return 0
    return face_height / face_width


def extract_features_from_frame(frame):
    gray, equalized = preprocess_frame(frame)
    faces = detector(equalized, 0)

    if len(faces) == 0:
        return None

    face  = max(faces, key=lambda r: r.width() * r.height())
    shape = predictor(equalized, face)
    shape = face_utils.shape_to_np(shape)

    leftEye  = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    mouth    = shape[mStart:mEnd]

    left_ear  = eye_aspect_ratio(leftEye)
    right_ear = eye_aspect_ratio(rightEye)
    avg_ear   = (left_ear + right_ear) / 2.0
    mar       = mouth_aspect_ratio(mouth)
    tilt      = head_tilt_angle(shape)
    nod       = head_nod_ratio(shape)

    return {
        "EAR":       round(avg_ear,   4),
        "MAR":       round(mar,       4),
        "tilt":      round(tilt,      4),
        "nod_ratio": round(nod,       4),
        "left_ear":  round(left_ear,  4),
        "right_ear": round(right_ear, 4),
        "shape":     shape,
        "face":      face
    }


def annotate_frame(frame, features, ear_threshold=0.20, mar_threshold=0.60):
    if features is None:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    shape    = features["shape"]
    leftEye  = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    mouth    = shape[mStart:mEnd]

    ear_color = (0, 0, 255) if features["EAR"] < ear_threshold else (0, 255, 0)
    mar_color = (0, 0, 255) if features["MAR"] > mar_threshold else (0, 255, 0)

    cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, ear_color, 1)
    cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, ear_color, 1)
    cv2.drawContours(frame, [cv2.convexHull(mouth)],    -1, (255, 255, 0), 1)

    cv2.putText(frame, "EAR: " + str(round(features["EAR"], 2)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_color, 2)
    cv2.putText(frame, "MAR: " + str(round(features["MAR"], 2)), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, mar_color, 2)
    cv2.putText(frame, "Tilt: " + str(round(features["tilt"], 1)), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

    return frame