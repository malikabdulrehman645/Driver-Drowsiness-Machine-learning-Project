# NOVA — AI-Powered Wearable Mobility Assistant

Research prototype combining camera perception, smart-cane sensing, context-aware priority management (CAPME), and a native Android companion app.

> **Status:** Initial development scaffold. Distance and user-context values in demos are simulated. Not a validated mobility or safety device.

## Quick start (Python 3.10+)

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
python scripts/demo_capme.py
python scripts/run_camera.py  # optional: install requirements-vision.txt first
```

## Components
- `ai/`: camera and object detection; future spatial estimation
- `capme/`: initial explainable prioritization baseline
- `hardware/`: ESP32 cane/wristband and Raspberry Pi placeholders
- `mobile/`: Kotlin Android starter instructions (create Android Studio project here)
- `communication/`: shared message schema, future BLE/Wi-Fi transport
- `experiments/`: fixed-threshold baseline and evaluation plans
- `docs/`: architecture, research, interface contracts and safety

## Important limits
- CAPME scores and thresholds are **illustrative research starting points**, not validated safety limits.
- The initial camera demo performs **object detection only**, not reliable depth or collision prediction.
- Never use this prototype as a substitute for a white cane, orientation/mobility training, or human assistance.

## Team workflow
Create feature branches (`feature/ai-camera`, `feature/capme`, `feature/android`, `feature/cane`), open pull requests, and run `pytest` before merging.
