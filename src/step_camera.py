"""
Este módulo captura imágenes de una cámara, procesa los frames y utiliza un modelo de PyTorch
para realizar predicciones basadas en el lenguaje de señas colombiano.
"""

import time
import cv2  # pylint: disable=no-member
import mediapipe as mp
import torch
import torch.nn.functional as F

# Cargar el modelo de PyTorch
model = torch.load("senalizaV2.pt", map_location=torch.device("cpu"))
model.eval()

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)  # pylint: disable=no-member

_, frame = cap.read()

H, W, C = frame.shape

IMG_COUNTER = 0
ANALYSIS_FRAME = ""
LETTER_PRED = [
    "A", "B", "C", "D", "E", "F", "H", "I", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "T", "U", "V", "W", "X", "Y"
]

while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)  # pylint: disable=no-member
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    if k % 256 == 32:
        ANALYSIS_FRAME = frame
        showframe = ANALYSIS_FRAME
        cv2.imshow("Frame", showframe)  # pylint: disable=no-member
        framergbanalysis = cv2.cvtColor(ANALYSIS_FRAME, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
        if hand_landmarksanalysis:
            for handLMsanalysis in hand_landmarksanalysis:
                X_MAX = 0
                Y_MAX = 0
                X_MIN = W
                Y_MIN = H
                for lmanalysis in handLMsanalysis.landmark:
                    x, y = int(lmanalysis.x * W), int(lmanalysis.y * H)
                    if x > X_MAX:
                        X_MAX = x
                    if x < X_MIN:
                        X_MIN = x
                    if y > Y_MAX:
                        Y_MAX = y
                    if y < Y_MIN:
                        Y_MIN = y
                Y_MIN -= 20
                Y_MAX += 20
                X_MIN -= 20
                X_MAX += 20

        ANALYSIS_FRAME = cv2.cvtColor(ANALYSIS_FRAME, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member
        ANALYSIS_FRAME = ANALYSIS_FRAME[Y_MIN:Y_MAX, X_MIN:X_MAX]
        ANALYSIS_FRAME = cv2.resize(ANALYSIS_FRAME, (64, 64))  # pylint: disable=no-member

        # Preprocesamiento actualizado para convertir de 1 a 3 canales
        ANALYSIS_FRAME = (
            torch.tensor(ANALYSIS_FRAME, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        ANALYSIS_FRAME /= 255.0

        # Convertir de 1 canal (escala de grises) a 3 canales
        ANALYSIS_FRAME = ANALYSIS_FRAME.repeat(1, 3, 1, 1)

        # Realizar predicción
        with torch.no_grad():
            prediction = model(ANALYSIS_FRAME)
            # Aplicar softmax si no está en el modelo
            prediction = F.softmax(prediction, dim=1)

        predarray = prediction.numpy()[0]
        letter_prediction_dict = {
            LETTER_PRED[i]: predarray[i] for i in range(len(LETTER_PRED))
        }
        predarrayordered = sorted(predarray, reverse=True)
        high1 = predarrayordered[0]
        high2 = predarrayordered[1]
        high3 = predarrayordered[2]
        for key, value in letter_prediction_dict.items():
            if value == high1:
                print("Predicted Character 1: ", key)
                print("Confidence 1: ", 100 * value)
            elif value == high2:
                print("Predicted Character 2: ", key)
                print("Confidence 2: ", 100 * value)
            elif value == high3:
                print("Predicted Character 3: ", key)
                print("Confidence 3: ", 100 * value)
        time.sleep(5)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            X_MAX = 0
            Y_MAX = 0
            X_MIN = W
            Y_MIN = H
            for lm in handLMs.landmark:
                x, y = int(lm.x * W), int(lm.y * H)
                if x > X_MAX:
                    X_MAX = x
                if x < X_MIN:
                    X_MIN = x
                if y > Y_MAX:
                    Y_MAX = y
                if y < Y_MIN:
                    Y_MIN = y
            Y_MIN -= 20
            Y_MAX += 20
            X_MIN -= 20
            X_MAX += 20
            cv2.rectangle(frame, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 255, 0), 2)  # pylint: disable=no-member
    cv2.imshow("Frame", frame)  # pylint: disable=no-member

cap.release()
cv2.destroyAllWindows()  # pylint: disable=no-member
