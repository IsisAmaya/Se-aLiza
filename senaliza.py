"""
Este es un módulo Flask para procesar y servir imágenes de video con OpenCV.
"""

import cv2
from flask import Flask, Response, render_template, redirect, url_for
from src.senaliza_v2 import *
from src.camara_flask import (hands_global, initialize_camera, letter_pred_global, model_global, process_frame)

app = Flask(__name__)
# Inicializar cámara

def generate_frames():
    """
    Genera frames de video en tiempo real desde la cámara y los procesa.
    """
    cap = initialize_camera()
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Procesar frame y hacer predicción
        frame = process_frame(frame, model_global, hands_global, letter_pred_global)

        # Codificar la imagen como JPEG para enviar al cliente
        ret, buffer = cv2.imencode(".jpg", frame) # pylint: disable=no-member
        frame = buffer.tobytes()

        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


# Rutas - (argumentos: Url) - Función
@app.route("/")
def inicio():
    """
    Renderiza la página de inicio.
    """
    return render_template("index.html")

# Rutas - (argumentos: Url) - Función
@app.route("/alphabet")
def alphabet():
    """
    Renderiza la página de inicio.
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


@app.route('/start_translate')
def start_translate():
    return redirect(url_for('alphabet'))

@app.route('/finish_translate')
def finish_translate():
    return redirect(url_for('inicio'))

# Iniciar app
if __name__ == "__main__":
    app.run("127.0.0.1", 5000, debug=True)
