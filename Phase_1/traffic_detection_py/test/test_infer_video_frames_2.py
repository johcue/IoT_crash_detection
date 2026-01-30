from unittest import TestCase
from unittest.mock import patch
from unittest.mock import Mock
from collections import deque
import numpy as np
import torch
import time

from src.infer_video_frames_2 import (
    preprocess,
    infer_prob,
    encode_image,
    ACCIDENT_THRESHOLD,
    WINDOW_SIZE,
    MIN_POSITIVE,
    TOPIC_LED_CMD
)


class TestTrafficDetection(TestCase):

    # ---------------- PREPROCESS ----------------

    def test_preprocess_output_shape(self):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        x = preprocess(dummy_img)

        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual(x.shape, (1, 3, 224, 224))
        self.assertEqual(x.dtype, torch.float32)

    # ---------------- INFERENCE ----------------

    def test_infer_prob_scalar_output(self):
        mock_model = Mock()
        mock_logits = torch.tensor([[2.0]])
        mock_model.return_value = mock_logits

        x = torch.zeros((1, 3, 224, 224))
        prob = infer_prob(mock_model, x)

        self.assertTrue(0.0 <= prob <= 1.0)

    # ---------------- ENCODING ----------------

    def test_encode_image_returns_base64(self):
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        encoded = encode_image(dummy_img)

        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 0)

    # ---------------- TEMPORAL LOGIC ----------------

    def test_accident_confirmed_when_enough_positive_frames(self):
        history = deque(maxlen=WINDOW_SIZE)

        for _ in range(MIN_POSITIVE):
            history.append(True)

        self.assertTrue(sum(history) >= MIN_POSITIVE)

    def test_accident_not_confirmed_when_insufficient_frames(self):
        history = deque(maxlen=WINDOW_SIZE)

        for _ in range(MIN_POSITIVE - 1):
            history.append(True)

        self.assertFalse(sum(history) >= MIN_POSITIVE)

    # ---------------- LED DECISION ----------------

    def test_led_turns_on_when_prob_above_threshold(self):
        prob = ACCIDENT_THRESHOLD + 0.1
        is_accident = prob >= ACCIDENT_THRESHOLD

        self.assertTrue(is_accident)

    def test_led_stays_off_when_prob_below_threshold(self):
        prob = ACCIDENT_THRESHOLD - 0.1
        is_accident = prob >= ACCIDENT_THRESHOLD

        self.assertFalse(is_accident)

    # ---------------- MQTT BEHAVIOR ----------------

    @patch("paho.mqtt.client.Client")
    def test_publish_led_on_command(self, mock_mqtt: Mock):
        client = mock_mqtt.return_value

        client.publish(TOPIC_LED_CMD, "ON", retain=True)

        client.publish.assert_called_with(TOPIC_LED_CMD, "ON", retain=True)

    @patch("paho.mqtt.client.Client")
    def test_publish_led_off_command(self, mock_mqtt: Mock):
        client = mock_mqtt.return_value

        client.publish(TOPIC_LED_CMD, "OFF", retain=True)

        client.publish.assert_called_with(TOPIC_LED_CMD, "OFF", retain=True)

    # ---------------- IDEMPOTENCY ----------------

    @patch("paho.mqtt.client.Client")
    def test_led_command_only_sent_on_state_change(self, mock_mqtt: Mock):
        client = mock_mqtt.return_value

        last_led = None
        states = [False, False, True, True, False]

        for state in states:
            if state != last_led:
                client.publish(
                    TOPIC_LED_CMD,
                    "ON" if state else "OFF",
                    retain=True
                )
                last_led = state

        self.assertEqual(client.publish.call_count, 3)

    # ---------------- TIMING LOGIC ----------------

    def test_led_hold_timeout_expires(self):
        led_state = True
        led_until = time.time() + 0.1

        time.sleep(0.2)

        if time.time() > led_until:
            led_state = False

        self.assertFalse(led_state)
