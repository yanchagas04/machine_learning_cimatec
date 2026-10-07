from src.engine import FaceRecognizer
import os

# Caminhos absolutos para as pastas do dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_DIR = os.path.join(BASE_DIR, "data", "auth")

if __name__ == "__main__":
    face_recognizer = FaceRecognizer(AUTH_DIR)
    face_recognizer.VerificationLoop()