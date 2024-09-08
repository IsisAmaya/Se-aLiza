"""
Este módulo captura video de la cámara, procesa los frames usando Mediapipe y un modelo de PyTorch para predecir letras en lenguaje de señas.
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

# Cargar el modelo de PyTorch
model = torch.load(
    r"D:\Universidad\Septimo Semestre\Proyecto Integrador II\proyecto\SenaLiza\src\senalizaV2.pt",
    map_location=torch.device("cpu")
)
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

letterpred = [
    "A", "B", "C", "D", "E", "F", "H", "I", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "T", "U", "V", "W", "X", "Y"
]

def initialize_camera():
    """Inicializa la cámara."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # pylint: disable=no-member
    return cap

def extract_hand_landmarks(hand_landmarks, frame_width, frame_height):
    """Extrae los límites de las landmarks de la mano."""
    x_max, y_max, x_min, y_min = 0, 0, frame_width, frame_height
    for landmark in hand_landmarks.landmark:
        x_coord, y_coord = int(landmark.x * frame_width), int(landmark.y * frame_height)
        x_max, x_min = max(x_coord, x_max), min(x_coord, x_min)
        y_max, y_min = max(y_coord, y_max), min(y_coord, y_min)
    return x_min, x_max, y_min, y_max

def predict_letter(loaded_model, analysis_frame, letter_pred):
    """Hace la predicción de la letra."""
    with torch.no_grad():
        prediction = loaded_model(analysis_frame)
        prediction = F.softmax(prediction, dim=1)
        predarray = prediction.numpy()[0]
        return letter_pred[np.argmax(predarray)]

def process_frame(frame, hands_processor):
    """Procesa el frame capturado y devuelve la predicción."""
    height, width, _ = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
    result = hands_processor.process(framergb)
    hand_landmarks_list = result.multi_hand_landmarks

    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            x_min, x_max, y_min, y_max = extract_hand_landmarks(hand_landmarks, width, height)

            analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member
            analysisframe = analysisframe[y_min:y_max, x_min:x_max]
            analysisframe = cv2.resize(analysisframe, (64, 64))  # pylint: disable=no-member
