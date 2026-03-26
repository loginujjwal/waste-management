import numpy as np
import random

class WasteClassifier:
    def __init__(self):
        self.classes = ["Organic", "Recyclable", "Hazardous"]

    def preprocess(self, image_array):
        return image_array / 255.0

    def predict(self, image_array):
        probs = np.random.rand(3)
        probs = probs / np.sum(probs)
        idx = np.argmax(probs)
        return self.classes[idx], probs[idx] * 100
