import sys
import os
import pytest
import torch
import cv2  # pylint: disable=no-member

# Asegúrate de que el path de la carpeta 'src' está agregado correctamente
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importar funciones necesarias desde camara_flask
from src.camara_flask import (hands_global, initialize_camera, letter_pred_global, process_frame)

@pytest.fixture
def model_fixture():
    """Cargar el modelo con los pesos para las pruebas sin deserializar el modelo completo."""
    # Importar la arquitectura correcta para los pesos (ejemplo: ColombianHandGestureResnet)
    from src.senaliza_v2 import ColombianHandGestureResnet  # Asegúrate de usar la arquitectura correcta

    # Inicializar el modelo
    model = ColombianHandGestureResnet()

    # Cargar solo los pesos del archivo .pth en lugar de deserializar el modelo completo
    state_dict = torch.load(
        r"C:\Users\rozos\OneDrive\Escritorio\SenaLiza\data\senalizaV5-1.pth",
        map_location=torch.device("cpu")  # Cargar en CPU
    )
    model.load_state_dict(state_dict)  # Cargar los pesos en el modelo

    # Cambiar el modelo a modo evaluación
    model.eval()

    return model

def test_initialize_camera():
    """Prueba la inicialización de la cámara."""
    cap = initialize_camera()
    assert cap.isOpened(), "La cámara no se ha inicializado correctamente."
    cap.release()  # Asegúrate de liberar la cámara después del test

def test_performance(model_fixture):
    """Prueba de rendimiento del procesamiento de un frame."""
    cap = initialize_camera()
    ret, frame = cap.read()
    assert ret, "No se pudo leer el frame de la cámara."

    # Medir el tiempo de procesamiento
    import time
    start_time = time.time()
    processed_frame = process_frame(frame, model_fixture, hands_global, letter_pred_global)
    end_time = time.time()

    processing_time = end_time - start_time
    cap.release()

    # Comprobar que el procesamiento toma menos de 1 segundo
    assert processing_time < 1, f"El tiempo de procesamiento es demasiado alto: {processing_time:.2f} segundos"

def test_prediction(model_fixture):
    """Prueba que el modelo realice una predicción sobre un frame."""
    cap = initialize_camera()
    ret, frame = cap.read()
    assert ret, "No se pudo leer el frame de la cámara."

    # Procesar frame
    processed_frame = process_frame(frame, model_fixture, hands_global, letter_pred_global)

    # Verificar que el frame procesado no es nulo
    assert processed_frame is not None, "El frame procesado es nulo."
    cap.release()
