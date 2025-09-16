from deepface import DeepFace
from predictor import Predictor


class VideoPredictor(Predictor):
    def __init__(self):
        super().__init__() 
        
    def predict_emotion(self, data):
        result = DeepFace.analyze(data, enforce_detection=False, actions=["emotion"])
        dominant_emotion = result[0]["dominant_emotion"]
        probabilities = result[0]["emotion"]
        return dominant_emotion, probabilities
