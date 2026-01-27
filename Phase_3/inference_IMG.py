import cv2
import torch
import numpy as np
import paho.mqtt.client as mqtt

IMAGE_PATH = "../test/images/test.jpg"
MODEL_PATH = "../Phase_2/models/accident_classifier_MobileNET.ts"

MQTT_HOST = ""          
MQTT_PORT = 1883
MQTT_TOPIC = "cv/accident/prob"

IMG_SIZE = 224

def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))          # [3,H,W]
    x = torch.from_numpy(x).unsqueeze(0)    # [1,3,H,W]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def load_model(path):
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model


def infer_prob(model, x):
    with torch.no_grad():
        logits = model(x)

        # handle either [1,1] or [1]
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]

        prob = torch.sigmoid(logits).item()
    return float(prob)


def main():
    # load image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"image not found: {IMAGE_PATH}")

    
    model = load_model(MODEL_PATH)

   
    x = preprocess(img)
    prob = infer_prob(model, x)

    print(f"prob_accident = {prob:.3f}")


    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_HOST, MQTT_PORT)
    client.publish(MQTT_TOPIC, f"{prob:.3f}")
    client.disconnect()


if __name__ == "__main__":
    main()

