import numpy as np
import tensorflow as tf

LABELS = ["ALERT", "DROWSY", "CRITICAL"]

class Predictor:
    def __init__(self, model_path, sequence_length):
        # Force tf.keras loader
        self.model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )
        self.sequence_length = sequence_length
        self.buffer = []

    def update(self, ear):
        self.buffer.append([ear])

        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)

        if len(self.buffer) < self.sequence_length:
            return None

        x = np.array(self.buffer).reshape(1, self.sequence_length, 1)
        probs = self.model.predict(x, verbose=0)[0]
        return LABELS[int(np.argmax(probs))]
