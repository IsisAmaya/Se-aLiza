"""
Este es un módulo Flask para procesar y servir imágenes de video con OpenCV.
"""

import cv2
import os
from flask import Flask, Response, render_template, redirect, url_for
from src.senaliza_v2 import *
from src.camara_flask import (
    hands_global,
    initialize_camera,
    letter_pred_global,
    model_global,
    process_frame,
)

app = Flask(__name__)

# Inicializar la cámara y generar frames
def generate_frames():
    """
    Genera frames de video en tiempo real desde la cámara y los procesa.
    """
    cap = initialize_camera()
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Procesar el frame y realizar predicción
        frame = process_frame(frame, model_global, hands_global, letter_pred_global)

        # Codificar la imagen como JPEG para enviar al cliente
        ret, buffer = cv2.imencode(".jpg", frame)  # pylint: disable=no-member
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

# Definición de rutas
@app.route("/")
def index():
    """
    Renderiza la página de inicio.
    """
    return render_template("index.html")

@app.route("/alphabet")
def alphabet():
    """
    Renderiza la página del alfabeto para la traducción.
    """
    return render_template("alphabet.html")

@app.route("/video_feed")
def video_feed():
    """
    Proporciona la transmisión de video procesada.
    """
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/start_translate")
def start_translate():
    """
    Redirige a la página del alfabeto para comenzar la traducción.
    """
    return redirect(url_for("alphabet"))

@app.route("/finish_translate")
def finish_translate():
    """
    Redirige de vuelta a la página de inicio al finalizar la traducción.
    """
    return redirect(url_for("index"))

@app.route('/biblioteca')
def biblioteca():
    gifs_folder = os.path.join(app.static_folder, 'GIFS')
    gifs = [f for f in os.listdir(gifs_folder) if f.endswith('.gif')]

    # Filtra los GIFs en alfabeto y palabras
    alfabeto = [gif for gif in gifs if len(gif.split('.')[0]) == 1]  # Letras
    palabras = [gif for gif in gifs if len(gif.split('.')[0]) > 1]  # Palabras

    return render_template('biblioteca.html', alfabeto=alfabeto, palabras=palabras)

# Iniciar la aplicación Flask
if __name__ == "__main__":
    app.run("127.0.0.1", 5000, debug=True)
