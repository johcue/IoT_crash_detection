class FakeMQTTClient:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload, retain=False):
        self.published.append((topic, payload, retain))

    def connect(self, *args, **kwargs):
        pass

    def disconnect(self):
        pass
