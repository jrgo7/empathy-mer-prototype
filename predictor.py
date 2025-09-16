from abc import ABC, abstractmethod
import os
from time import time
import logging
import pprint
from pathlib import Path

class Predictor(ABC):

    def test_predict_emotion(self, test_folder: Path) -> None:
        """
        Test the predict_emotion function on files in test_folder.
        """
        for file_name in os.listdir(test_folder):
            logging.info(f"Predicting emotion of {file_name}")
            
            prev = time()
            dominant_emotion, probabilities = self.predict_emotion(Path(test_folder, file_name))
            cur = time()
            diff = prev - cur

            logging.info(f"{dominant_emotion=}")
            pprint.pprint(probabilities)
            
            logging.info(f"Testing took {round(diff)} seconds\n\n")
            
    @abstractmethod
    def predict_emotion(self, data) -> tuple[str, dict]:
        """
        Analyze `data` and return a tuple (dominant_emotion, probabilities)
        where probabilities contains a probability distribution for each emotion
        """
        pass