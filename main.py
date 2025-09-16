import cv2 as cv
import pyaudio
from cv2_enumerate_cameras import enumerate_cameras
from tabulate import tabulate

from video_predictor import VideoPredictor
from audio_predictor import AudioPredictor

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)


def select_camera() -> int:
    """
    Select a camera to use.
    """
    camera_info_headers = ["Index", "Camera Name", "Path"]
    camera_info = [
        [camera.index, camera.name, camera.path] for camera in enumerate_cameras()
    ]
    print(tabulate(camera_info, headers=camera_info_headers))
    return int(input(">>> Select camera index: "))


def display_result(
    frame: cv.UMat,
    result: tuple[str, dict],
    position: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """
    Convenience method to display model prediction results
    """
    dominant_emotion, emotion_probabilities = result
    x, y = position
    dy = 15
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.5

    cv.putText(frame, dominant_emotion, (x, y), font, scale, color)

    for emotion, probabilty in emotion_probabilities.items():
        y += dy
        text = f"{emotion}: {round(float(probabilty), 2)}"
        cv.putText(frame, text, (x, y), font, scale, color)


def combine_emotion(video_result: dict, audio_result: dict) -> tuple[str, dict]:
    """
    Combine the video (face) and audio (voice) results.
    Audio and video are weighted 0.4 and 0.6 respectively.
    """
    FACE_TO_AUDIO_MAP = {
        "fear": "fearful",
        "surprise": "surprised",
        "angry": "angry",
        "disgust": "disgust",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
    }
    ALL_AUDIO_EMOTIONS = FACE_TO_AUDIO_MAP.values()

    local_probabilities = {key: 0.0 for key in ALL_AUDIO_EMOTIONS}

    # Apply audio probabilities (weighted by 0.4)
    for emotion, prob in audio_result.items():
        if emotion in local_probabilities:
            local_probabilities[emotion] = prob * 0.4

    # Apply face probabilities (mapped and weighted by 0.6)
    for face_emotion, prob in video_result.items():
        audio_key = FACE_TO_AUDIO_MAP.get(face_emotion)
        if audio_key is not None:
            local_probabilities[audio_key] += prob * 0.6
        else:
            print(f"Unmapped face emotion: '{face_emotion}' (no matching audio label)")

    dominant_emotion = max(local_probabilities, key=local_probabilities.get)
    return dominant_emotion, local_probabilities


def main() -> None:
    # Initialize video capture
    camera_index = select_camera()
    video_capture = cv.VideoCapture(camera_index)

    # Initialize audio capture
    CHUNK = 1024
    FS = 16000  # sample rate
    SECONDS = 3
    p = pyaudio.PyAudio()
    audio_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=FS,
        frames_per_buffer=CHUNK,
        input=True,
    )

    # Initalize predictors and results
    video_predictor = VideoPredictor()
    audio_predictor = AudioPredictor()
    video_result = None
    audio_result = None
    fused_result = None

    # Main loop
    tick = 0
    audio_frames = []
    is_quit = False
    while not is_quit:
        # Capture
        _, video_frame = video_capture.read()
        audio_frames.append(audio_stream.read(CHUNK, exception_on_overflow=False))

        # Analyze
        if tick % 15 == 0:
            video_result = video_predictor.predict_emotion(video_frame)

        if len(audio_frames) == int(FS / CHUNK * SECONDS):
            audio_result = audio_predictor.predict_emotion(audio_frames)
            audio_frames = []

        # Display
        if video_result:
            display_result(video_frame, video_result, (0, 32), COLOR_RED)

        if audio_result:
            display_result(video_frame, audio_result, (150, 32), COLOR_GREEN)

        if video_result and audio_result:
            fused_result = combine_emotion(video_result, audio_result)
            display_result(video_frame, fused_result, (300, 32), COLOR_BLUE)

        cv.imshow("Empathy S13 G4 Prototype", video_frame)

        # Quit the program when we press 'q'
        if cv.waitKey(1) == ord("q"):
            is_quit = True

        tick += 1


if __name__ == "__main__":
    main()
