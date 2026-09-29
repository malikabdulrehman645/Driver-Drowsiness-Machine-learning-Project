import streamlit as st
import cv2
import numpy as np
import joblib
import time
import pandas as pd
import plotly.express as px
from imutils import face_utils

from feature_extraction import (
    eye_aspect_ratio, mouth_aspect_ratio, head_tilt_angle, head_nod_ratio,
    detector, predictor, lStart, lEnd, rStart, rEnd, mStart, mEnd
)
from preprocessing import preprocess_frame

st.set_page_config(page_title="Drowsiness Detection", page_icon="🚗", layout="wide")
st.title("🚗 Driver Drowsiness Detection System")

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.header("Settings")
ear_thresh   = st.sidebar.slider("EAR Threshold",        0.10, 0.35, 0.20, 0.01)
mar_thresh   = st.sidebar.slider("MAR Threshold",        0.40, 0.80, 0.60, 0.01)
alert_frames = st.sidebar.slider("Alert after N frames", 5,    40,   20)
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression", "SVM"])


@st.cache_resource
def load_model(name):
    path   = name.replace(" ", "_").lower()
    model  = joblib.load(f"models/{path}.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


model, scaler = load_model(model_choice)

# ── Layout ───────────────────────────────────────────────────────────
col_video, col_info = st.columns([2, 1])
video_box   = col_video.empty()
status_box  = col_info.empty()
metrics_box = col_info.empty()
chart_box   = col_info.empty()

# ── Session state ────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "log" not in st.session_state:
    st.session_state.log = {"t": [], "EAR": [], "MAR": [], "drowsy": []}

b1, b2 = st.columns(2)
if b1.button("▶  Start", use_container_width=True):
    st.session_state.running = True
    st.session_state.log     = {"t": [], "EAR": [], "MAR": [], "drowsy": []}
if b2.button("⏹  Stop",  use_container_width=True):
    st.session_state.running = False

# ── Detection loop ───────────────────────────────────────────────────
if st.session_state.running:
    cap = cv2.VideoCapture(0)

    eye_counter  = 0
    yawn_counter = 0
    blink_count  = 0
    was_closed   = False
    t0           = time.time()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot open webcam.")
            break

        frame           = cv2.flip(frame, 1)
        gray, equalized = preprocess_frame(frame)
        faces           = detector(equalized, 0)

        is_drowsy = False
        avg_ear   = 0.0
        mar       = 0.0
        prob      = 0.0

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

            if avg_ear < ear_thresh:
                eye_counter += 1
                was_closed   = True
            else:
                if was_closed and eye_counter >= 2:
                    blink_count += 1
                was_closed  = False
                eye_counter = 0

            yawn_counter = yawn_counter + 1 if mar > mar_thresh else 0

            fv        = scaler.transform([[avg_ear, mar, tilt, nod, left_ear, right_ear]])
            pred      = model.predict(fv)[0]
            prob      = model.predict_proba(fv)[0][1]
            is_drowsy = pred == 1 or eye_counter >= alert_frames or yawn_counter >= 15

            color = (0, 0, 255) if is_drowsy else (0, 200, 0)
            cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, color, 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, color, 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)],    -1, (255, 220, 0), 1)

            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        label = "DROWSY!" if is_drowsy else "ALERT"
        color = (0, 0, 255) if is_drowsy else (0, 200, 0)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (20, 20, 20), -1)
        cv2.putText(frame, f"STATUS: {label}", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        video_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB", use_column_width=True)

        elapsed = round(time.time() - t0, 1)
        st.session_state.log["t"].append(elapsed)
        st.session_state.log["EAR"].append(round(avg_ear, 3))
        st.session_state.log["MAR"].append(round(mar, 3))
        st.session_state.log["drowsy"].append(1 if is_drowsy else 0)

        bg = "#cc0000" if is_drowsy else "#007700"
        status_box.markdown(
            f'<div style="background:{bg};color:#fff;padding:14px;'
            f'border-radius:8px;text-align:center;font-size:22px;'
            f'font-weight:bold;margin-bottom:10px">{label}</div>',
            unsafe_allow_html=True)

        with metrics_box.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("EAR",    f"{avg_ear:.3f}")
            c2.metric("MAR",    f"{mar:.3f}")
            c3.metric("Blinks", blink_count)
            st.metric("Drowsy probability", f"{prob:.1%}")

        if len(st.session_state.log["t"]) > 3:
            df_log = pd.DataFrame(st.session_state.log).tail(60)
            fig    = px.line(df_log, x="t", y="EAR",
                             labels={"t": "Time (s)"}, height=180)
            fig.add_hline(y=ear_thresh, line_dash="dash", line_color="red")
            fig.update_layout(margin=dict(t=10, b=10, l=0, r=0), showlegend=False)
            chart_box.plotly_chart(fig, use_container_width=True)

    cap.release()

# ── Session analytics ────────────────────────────────────────────────
if st.session_state.log["t"]:
    st.divider()
    st.subheader("Session Summary")
    df_full = pd.DataFrame(st.session_state.log)
    a, b, c = st.columns(3)
    a.metric("Avg EAR",  f"{df_full['EAR'].mean():.3f}")
    b.metric("Avg MAR",  f"{df_full['MAR'].mean():.3f}")
    c.metric("Drowsy %", f"{df_full['drowsy'].mean():.1%}")
    fig2 = px.line(df_full, x="t", y=["EAR", "MAR"],
                   labels={"t": "Time (s)", "value": "Score"})
    st.plotly_chart(fig2, use_container_width=True)