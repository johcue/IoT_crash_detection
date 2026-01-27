import cv2
import torch
import numpy as np
import paho.mqtt.client as mqtt
import base64
import argparse
import os
import time
from collections import deque

MODEL_PATH = "../Phase_2/models/accident_classifier_MobileNET.ts"

MQTT_HOST = ""
MQTT_PORT = 1883

TOPIC_PROB = "cv/accident/prob"
TOPIC_IMAGE = "cv/frame/image"
TOPIC_STATUS = "cv/system/status"
TOPIC_LED_CMD = "iot/led/cmd"

IMG_SIZE = 224
JPEG_QUALITY = 70

ACCIDENT_THRESHOLD = 0.8

WINDOW_SIZE = 10  
MIN_POSITIVE = 7  

LED_HOLD_SECONDS = 5  
FRAME_DELAY = 0.1  

POST_VIDEO_HOLD_SECONDS = 5


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    return (x - mean) / std


def load_model(path):
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model


def infer_prob(model, x):
    with torch.no_grad():
        logits = model(x)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        return float(torch.sigmoid(logits).item())


def encode_image(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")


def main():
   
    parser = argparse.ArgumentParser(
        description="Video-based accident detection with debounced LED control"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

   
    model = load_model(MODEL_PATH)

   
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_HOST, MQTT_PORT)

    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    
    history = deque(maxlen=WINDOW_SIZE)

    led_state = False
    led_until = 0.0
    last_published_led = None
    last_status = None

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        x = preprocess(frame)
        prob = infer_prob(model, x)

        is_accident = prob >= ACCIDENT_THRESHOLD
        history.append(is_accident)
        
        accident_confirmed = sum(history) >= MIN_POSITIVE
        now = time.time()

        if accident_confirmed:
            led_state = True
            led_until = now + LED_HOLD_SECONDS
        elif now > led_until:
            led_state = False

        status = "Accident detected" if led_state else "Normal traffic"

        client.publish(TOPIC_PROB, f"{prob:.3f}")
        img_b64 = encode_image(frame)
        if img_b64:
            client.publish(TOPIC_IMAGE, img_b64)

        if led_state != last_published_led:
            client.publish(TOPIC_LED_CMD, "ON" if led_state else "OFF", retain=True)
            last_published_led = led_state

        if status != last_status:
            client.publish(TOPIC_STATUS, status)
            last_status = status

        time.sleep(FRAME_DELAY)

    cap.release()
    print("\n--- Video finished ---")
    if led_state:
        print(
            f"Post-video alert hold: keeping LED ON for "
            f"{POST_VIDEO_HOLD_SECONDS} seconds"
        )

        time.sleep(POST_VIDEO_HOLD_SECONDS)
        print("Post-video shutdown: turning LED OFF")
        client.publish(TOPIC_LED_CMD, "OFF", retain=False)
    
    client.disconnect()
    print("--- System shutdown complete ---\n")


if __name__ == "__main__":
    main()
