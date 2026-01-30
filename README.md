# IoT Traffic Accident Detection and Alert System

This repository implements a video-based traffic accident detection system integrated with an IoT alert pipeline. The system combines deep learning, temporal decision logic, MQTT communication, and embedded firmware to provide stable, real-time accident alerts using both a physical LED (ESP32) and a Node-RED dashboard.

Developed as part of the **Embedded Systems course** at the **University of Salerno (January 2026).


---

## Demo

<p align="center">
  <video src="demo.mp4" controls width="700"></video>
</p>

> If the video does not render inline on GitHub,
> 👉 [Click here to download and watch the demo](demo.mp4)

The demo shows:

* Real-time video inference
* Stable accident confirmation
* MQTT communication
* ESP32 LED activation
* Node-RED dashboard synchronization

---

## Motivation

Video-based AI models operate at the **frame level**, producing noisy predictions.
Reacting to individual frames leads to:

* Rapid ON/OFF flickering
* False alarms
* Loss of operator trust
* Operational inefficiency

This project solves the issue using **temporal stabilization with a sliding window and voting logic**, mimicking how humans confirm events over time.

---

## System Architecture

```
Video → CNN → Temporal Logic → MQTT → ESP32 + Node-RED Dashboard
```

1. Video frames processed by a CNN (MobileNetV2)
2. Frame-level accident probabilities
3. Temporal stabilization (sliding window + voting)
4. MQTT-based event publishing
5. ESP32 reacts to commands
6. Visual alerts (LED + dashboard)

---

## Repository Structure

```
.
├── demo.mp4
├── Phase_1
│   ├── platformIO_test
│   └── traffic_detection_py
│       ├── mock
│       ├── src
│       └── test
├── Phase_2
│   └── train_mobileNet.ipynb
├── Phase_3
│   ├── inference_IMG.py
│   └── inference_VID.py
├── slides
│   ├── Embedded Systems Project.pdf
│   └── Embedded Systems Company.pdf
└── test
    ├── images
    └── videos
```

---

## Phase 1 — Test-Driven Foundations

**Goal:** Ensure all logic is **verifiable before integration**.

### Python (Inference Side)

* Image preprocessing
* Model wrapper
* Temporal voting logic
* LED state machine
* MQTT publishing behavior

All external dependencies (e.g., MQTT) are **mocked**.

### Embedded (ESP32)

* LED logic as a **pure function**
* Hardware isolated
* Tested using **PlatformIO native tests**

---

## Phase 2 — Traffic Accident Recognition

* **Model:** MobileNetV2
* **Dataset:** [Car Crash Dataset (CCD)](https://github.com/Cogito2012/CarCrashDataset)
* **Task:** Frame-level binary classification
* **Deployment:** Exported as TorchScript

---

## Phase 3 — Inference & IoT Integration

### Python Inference Pipeline

* OpenCV video/image input
* Preprocessing (224×224, normalization)
* TorchScript inference
* Sigmoid → probability
* MQTT publishing

### Temporal Stabilization

* Sliding window of last **N frames**
* Binary vote per frame
* Accident confirmed only if:

  * Positive votes ≥ threshold
* Enforced minimum LED ON duration
* Safe shutdown after video end

### ESP32 Firmware

* Subscribes to MQTT LED commands
* Ignores redundant messages
* Publishes retained LED state
* Reconnect-safe behavior

---

## Setup & Run Instructions

### 1 — Conda (Python Inference)

#### 1. Create environment

```bash
conda create -n traffic-iot python=3.10 -y
conda activate traffic-iot
```

#### 2. Install dependencies

```bash
pip install torch torchvision opencv-python numpy paho-mqtt
```

#### 3. Run video inference

```bash
python Phase_3/inference_VID.py --video test/videos/000007.mp4
```

#### 4. Run image inference (optional)

```bash
python Phase_3/inference_IMG.py --image test/images/test_crash.jpg
```

---

### 2 — Docker (MQTT + Node-RED)

This project assumes:

* **MQTT broker** (e.g. Mosquitto)
* **Node-RED dashboard**

#### 1. Start services

```bash
docker-compose up -d
```

(If you don’t have a compose file yet, use:

* `eclipse-mosquitto`
* `nodered/node-red`)

#### 2. Open Node-RED

```
http://localhost:1880
```

Import the dashboard flow and subscribe to the MQTT topics used by the inference scripts.

---

### ESP32 Firmware (PlatformIO)

1. Open `Phase_1/platformIO_test` in PlatformIO
2. Flash firmware to ESP32
3. Configure:
   * Wi-Fi credentials
   * MQTT broker address
4. ESP32 subscribes to LED command topic and reacts automatically

---


## Author

**Johan Chicue Garcia**
Embedded Systems Project — University of Salerno
