"""
Este módulo captura video de la cámara, procesa los frames usando Mediapipe y un modelo de PyTorch para predecir letras en lenguaje de señas.
"""

import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
from src.senaliza_v2 import *

# Cargar el modelo de PyTorch
model = torch.load(r'D:\Universidad\Septimo Semestre\Proyecto Integrador II\proyecto\SenaLiza\src\senalizaV5.pt', map_location=torch.device('cpu'))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']

def initialize_camera():
    """Inicializa la cámara."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cap

def process_frame(frame, model, hands, letterpred):
    """Procesa el frame capturado y devuelve la predicción."""
    h, w, _ = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max, y_max, x_min, y_min = 0, 0, w, h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max, x_min = max(x, x_max), min(x, x_min)
                y_max, y_min = max(y, y_max), min(y, y_min)

            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20

            analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            analysisframe = analysisframe[y_min:y_max, x_min:x_max]
            analysisframe = cv2.resize(analysisframe, (64, 64))
            analysisframe = torch.tensor(analysisframe, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            analysisframe = analysisframe.repeat(1, 3, 1, 1)

            with torch.no_grad():
                prediction = model(analysisframe)
                prediction = F.softmax(prediction, dim=1)
                predarray = prediction.numpy()[0]
                letter_prediction = letterpred[np.argmax(predarray)]

            # Dibujar predicción en el frame
            cv2.putText(frame, f'Pred: {letter_prediction}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame