import cv2
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from imutils import face_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

from feature_extraction import (
    eye_aspect_ratio, mouth_aspect_ratio, head_tilt_angle, head_nod_ratio,
    detector, predictor, lStart, lEnd, rStart, rEnd, mStart, mEnd
)
from preprocessing import preprocess_frame

os.makedirs("models", exist_ok=True)


def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray, equalized = preprocess_frame(img)
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

    return [avg_ear, mar, tilt, nod, left_ear, right_ear]


def build_dataset(dataset_path="dataset"):
    folders = {
        "Open_Eyes":   0,
        "Closed_Eyes": 1,
        "Yawn":        1,
        "no_yawn":     0
    }

    records = []

    for folder_name, label in folders.items():
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"WARNING: Folder not found — {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Processing {len(files)} images from {folder_name}...")

        for i, fname in enumerate(files):
            features = process_image(os.path.join(folder_path, fname))
            if features:
                records.append(features + [label])
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(files)} done")

    columns = ["EAR", "MAR", "tilt", "nod_ratio", "left_ear", "right_ear", "label"]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv("features_dataset.csv", index=False)
    print(f"\nDataset built: {len(df)} samples")
    print(df["label"].value_counts().rename({0: "Alert", 1: "Drowsy"}))
    return df


def train_models(df=None):
    if df is None:
        df = pd.read_csv("features_dataset.csv")

    X = df[["EAR", "MAR", "tilt", "nod_ratio", "left_ear", "right_ear"]]
    y = df["label"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42)
    }

    results = {}

    print("\n" + "="*55)
    print("MODEL RESULTS")
    print("="*55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)
        acc       = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        results[name] = {"model": model, "accuracy": acc, "y_pred": y_pred}

        print(f"\n{name}  —  Accuracy: {acc:.4f}  |  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(classification_report(y_test, y_pred, target_names=["Alert", "Drowsy"]))

    # Confusion matrix plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Alert", "Drowsy"],
                    yticklabels=["Alert", "Drowsy"],
                    ax=axes[idx])
        axes[idx].set_title(f"{name}\nAcc: {res['accuracy']:.3f}")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=120)
    plt.show()
    print("\nSaved: model_comparison.png")

    # Feature importance
    rf  = results["Random Forest"]["model"]
    imp = rf.feature_importances_
    plt.figure(figsize=(7, 4))
    plt.bar(["EAR", "MAR", "Tilt", "Nod", "L-EAR", "R-EAR"], imp, color="steelblue")
    plt.title("Feature Importance — Random Forest")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.show()
    print("Saved: feature_importance.png")

    # Save all models
    for name, res in results.items():
        fname = name.replace(" ", "_").lower()
        joblib.dump(res["model"], f"models/{fname}.pkl")

    joblib.dump(scaler, "models/scaler.pkl")

    best = max(results, key=lambda k: results[k]["accuracy"])
    joblib.dump(results[best]["model"], "models/best_model.pkl")
    print(f"\nBest model: {best} saved as best_model.pkl")

    return results, scaler


if __name__ == "__main__":
    print("=== BUILDING DATASET ===")
    df = build_dataset("dataset/train")
    print("\n=== TRAINING MODELS ===")
    train_models(df)
    print("\nAll done. Check models/ folder.")