from deepface import DeepFace
import cv2 as cv

# TODO: Analyze audio as well
# Either we get an speech emotion recognition library and fuse these two together
# Or we get something that detects emotion from both already (and replace this entirely)

def analyze_video(frame: cv.UMat) -> str:
    # Analyze a video frame and return the dominant emotion detected
    result = DeepFace.analyze(frame, enforce_detection=False, actions=["emotion"])
    dominant_emotion = result[0]["dominant_emotion"]
    return dominant_emotion


def put_text(frame: cv.UMat, text: str, position: tuple) -> None:
    # Convenience method to apply frame with text at a certain position
    cv.putText(frame, text, (0, 32), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))


def main():
    # Initialize video capture
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        emotion = analyze_video(frame)

        put_text(frame, emotion, (0, 32))

        cv.imshow("frame", frame)

        if cv.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
