"""
Este módulo captura video desde la cámara, procesa los frames utilizando Mediapipe y un modelo de PyTorch para predecir letras en lenguaje de señas colombianas.
"""

import torch
import torch.nn.functional as F
import cv2
import mediapipe as mp

# Cargar el modelo de PyTorch
model = torch.load(
    "senalizaV2.pt", 
    map_location=torch.device("cpu")
)
model.eval()

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)  # pylint: disable=no-member

_, frame = cap.read()
h, w, c = frame.shape

letterpred = [
    "A", "B", "C", "D", "E", "F", "H", "I", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "T", "U", "V", "W", "X", "Y"
]

while True:
    _, frame = cap.read()
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for hand_landmarks_set in hand_landmarks:
            x_max = 0  # pylint: disable=invalid-name
            y_max = 0  # pylint: disable=invalid-name
            x_min = w  # pylint: disable=invalid-name
            y_min = h  # pylint: disable=invalid-name
            for landmark in hand_landmarks_set.landmark:
                x_coord, y_coord = int(landmark.x * w), int(landmark.y * h)
                if x_coord > x_max:
                    x_max = x_coord
                if x_coord < x_min:
                    x_min = x_coord
                if y_coord > y_max:
                    y_max = y_coord
                if y_coord < y_min:
                    y_min = y_coord
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20

            analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member
            analysisframe = analysisframe[y_min:y_max, x_min:x_max]
            analysisframe = cv2.resize(analysisframe, (64, 64))  # pylint: disable=no-member

            # Preprocesamiento actualizado para convertir de 1 a 3 canales
            analysisframe = (
                torch.tensor(analysisframe, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            analysisframe /= 255.0
            analysisframe = analysisframe.repeat(1, 3, 1, 1)

            # Realizar predicción
            with torch.no_grad():
                prediction = model(analysisframe)
                prediction = F.softmax(prediction, dim=1)

            predarray = prediction.numpy()[0]
            letter_prediction_dict = {
                letterpred[i]: predarray[i] for i in range(len(letterpred))
            }
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]

            for key, value in letter_prediction_dict.items():
                if value == high1:
                    print(f"Predicted Character: {key} | Confidence: {100*value:.2f}%")
                    break

    cv2.imshow("Frame", frame)  # pylint: disable=no-member

    k = cv2.waitKey(1)  # pylint: disable=no-member
    if k % 256 == 27:  # Presionar 'ESC' para salir
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()  # pylint: disable=no-member
