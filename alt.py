from deepface import DeepFace
import cv2 as cv


from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np
import pyaudio
import threading 

# global vars for audio portion
model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
audio_model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = audio_model.config.id2label
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model.to(device)

# -- audio --
lock = threading.Lock() # i hate race conditions
audio_emotion = "nothing"
audio_probabilities = {}
def stream_audio():


    CHUNK = 1024
    SAMPLE_FORMAT = pyaudio.paInt16
    CHANNELS = 1
    FS = 16000 # sample rate
    SECONDS = 3

    p = pyaudio.PyAudio()  
    stream = p.open(format=SAMPLE_FORMAT, channels=CHANNELS, rate=FS, frames_per_buffer=CHUNK, input=True)

    while True:
        frames = []
        for _ in range(0, int(FS / CHUNK * SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        audio_data_np = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data_float = audio_data_np.astype(np.float32) / 32768.0

        predicted_emotion, raw_probabilities = audio_arr_emotion_analyzer(audio_data_float)
            
        with lock:
            global audio_emotion, audio_probabilities
            audio_emotion = predicted_emotion
            audio_probabilities = raw_probabilities


def audio_arr_emotion_analyzer(arr):
    # processing
    max_length = int(feature_extractor.sampling_rate * 30.0)
    if len(arr) > max_length:
        arr = arr[:max_length]
    else:
        arr = np.pad(arr, (0, max_length - len(arr)))

    inputs = feature_extractor(arr, sampling_rate=feature_extractor.sampling_rate, max_length=max_length, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = audio_model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    
    normalized_prob = torch.nn.functional.softmax(logits, dim=-1)
    scores = normalized_prob.squeeze().tolist()
    probabilities = {id2label[i]: score * 100 for i, score in enumerate(scores)} # Convert to percentage

    return id2label[predicted_id], probabilities

def analyze_video(frame: cv.UMat) -> str:
    # Analyze a video frame and return the dominant emotion detected
    result = DeepFace.analyze(frame, enforce_detection=False, actions=["emotion"])
    dominant_emotion = result[0]["dominant_emotion"]
    probabitilies = result[0]["emotion"]
    return dominant_emotion, probabitilies


def put_text(frame: cv.UMat, text: str, position: tuple) -> None:
    # Convenience method to apply frame with text at a certain position
    cv.putText(frame, text, (0, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    

FACE_TO_AUDIO_MAP = {
    "fear": "fearful",
    "surprise": "surprised",
    "angry": "angry",
    "disgust": "disgust",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad"
}
ALL_AUDIO_EMOTIONS = list(id2label.values())

def combine_emotion(face_probabilities: dict) -> tuple[str, dict]:
    local_probabilities = {key: 0.0 for key in ALL_AUDIO_EMOTIONS}
    
    # Apply audio probabilities (weighted by 0.4)
    for emotion, prob in audio_probabilities.items():
        if emotion in local_probabilities:
            local_probabilities[emotion] = prob * 0.4
    
    # Apply face probabilities (mapped and weighted by 0.6)
    for face_emotion, prob in face_probabilities.items():
        audio_key = FACE_TO_AUDIO_MAP.get(face_emotion)
        if audio_key is not None:
            local_probabilities[audio_key] += prob * 0.6
        else:
            print(f"Unmapped face emotion: '{face_emotion}' (no matching audio label)")

    dominant_emotion = max(local_probabilities, key=local_probabilities.get)
    return dominant_emotion, local_probabilities

def main():
    # Initialize video capture
    tick = 0
    cap = cv.VideoCapture(0)

    # Initialize audio
    audio_thread = threading.Thread(target=stream_audio)
    audio_thread.daemon = True
    audio_thread.start()

    print("Audio model labels:", list(id2label.values()))  # For debugging — remove later


    while True:
        ret, frame = cap.read()
        if tick % 15 == 0:
            emotion, probabilities = analyze_video(frame)

        # video
        inc = 15

        
        for emotion, emotion_probability in probabilities.items():
            text = f"{emotion}: {emotion_probability:.2f}%" 
            cv.putText(frame, text, (0, 32 + inc), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
            inc += 15
        put_text(frame, max(probabilities, key=probabilities.get), (0, 32))

        # audio
        inc = 15

        with lock:
            local_audio_emotion = audio_emotion
            local_audio_probabilities = audio_probabilities.copy()

    
        for emotion, emotion_probability in local_audio_probabilities.items():
            text = f"{emotion}: {emotion_probability:.2f}%" 
            cv.putText(frame, text, (150, 32 + inc), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))
            inc += 15
        cv.putText(frame, audio_emotion, (150, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        

        # combined
        inc = 15
        combined_emotions, combined_probabilities = combine_emotion(probabilities)

        for emotion, emotion_probability in combined_probabilities.items():
            text = f"{emotion}: {emotion_probability:.2f}%"
            cv.putText(frame, text, (300, 32 + inc), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0))
            inc += 15
        cv.putText(frame, combined_emotions, (300, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))



        cv.imshow("frame", frame)

        if cv.waitKey(1) == ord("q"):
            break
            
        tick += 1

if __name__ == "__main__":
    main()
