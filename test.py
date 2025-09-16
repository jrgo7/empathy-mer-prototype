from video_predictor import VideoPredictor
from audio_predictor import AudioPredictor
from pathlib import Path
import logging


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    video_predictor = VideoPredictor()
    audio_predictor = AudioPredictor()

    video_predictor.test_predict_emotion(Path("test_assets", "image"))

    # TODO: Let AudioPredictor.predict_emotion accept file paths;
    #       currently this doesn't work (anymore) due to how we process raw audio
    #       instead of taking in a file path.
    # audio_predictor.test_predict_emotion(Path("test_assets", "audio"))


if __name__ == "__main__":
    main()
