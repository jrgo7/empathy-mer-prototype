import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import numpy as np
from predictor import Predictor


class AudioPredictor(Predictor):
    def __init__(self):
        super().__init__() 

    model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    audio_model = AutoModelForAudioClassification.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id, do_normalize=True
    )
    id2label = audio_model.config.id2label
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model.to(device)

    def predict_emotion(self, data):
        # processing
        audio_data_np = np.frombuffer(b"".join(data), dtype=np.int16)
        arr = audio_data_np.astype(np.float32) / 32768.0

        max_length = int(self.feature_extractor.sampling_rate * 30.0)
        if len(arr) > max_length:
            arr = arr[:max_length]
        else:
            arr = np.pad(arr, (0, max_length - len(arr)))

        inputs = self.feature_extractor(
            arr,
            sampling_rate=self.feature_extractor.sampling_rate,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.audio_model(**inputs)

        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1).item()

        normalized_prob = torch.nn.functional.softmax(logits, dim=-1)
        scores = normalized_prob.squeeze().tolist()
        probabilities = {
            self.id2label[i]: score * 100 for i, score in enumerate(scores)
        }  # Convert to percentage

        return self.id2label[predicted_id], probabilities
