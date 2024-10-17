"""
Este es un módulo Flask para procesar y servir imágenes de video con OpenCV.
"""

import cv2
import os
import queue
import time
import threading
from datetime import datetime
from flask import Flask, Response, render_template, redirect, url_for, jsonify, session, send_file
from flask_socketio import SocketIO, emit
from src.senaliza_v2 import *
from src.camara_flask import (
    hands_global,
    initialize_camera,
    letter_pred_global,
    model_global,
    process_frame,
)
from src.recording import (
    process_recording,
)

app = Flask(__name__)
socketio = SocketIO(app)

# Variables globales
is_recording = False
frame_queue = queue.Queue()
prediction = ""

history_folder = "histories"
historial = []

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

def generate_video():
    """Genera frames de video en tiempo real para la transmisión en Flask."""
    global is_recording
    cap = initialize_camera()
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Captura de frames controlada
        if is_recording:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_queue.put(frame_rgb)

        frame_count += 1

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        # Enviar el frame al cliente
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

    cap.release()
    

def send_updates():
    global prediction
    while True:
        time.sleep(5)  # Actualizar cada 5 segundos
        dynamic_string = "La palabra es: " + prediction
        socketio.emit('update_string', {'text': dynamic_string})

def process_after_stop():
    """Función que procesa la grabación después de detenerla."""
    global prediction
    global historial
    if frame_queue:
        prediction = process_recording(frame_queue)
        now = datetime.now()
        dt = now.strftime("%d/%m/%Y %H:%M:%S")
        st = dt + " - " + prediction
        historial.append(st)
        print(f"Predicción: {prediction}")
    else:
        print("No hay datos en la cola para procesar.")

@socketio.on('connect')
def handle_connect():
    threading.Thread(target=send_updates).start()

@app.route("/start_recording", methods=["POST"])
def start_recording():
    """Inicia la grabación."""
    global is_recording
    is_recording = True
    print("Grabación iniciada")
    return jsonify({"message": "Grabación iniciada"})

@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    """Detiene la grabación y lanza el procesamiento en segundo plano."""
    global is_recording
    is_recording = False
    print("Grabación detenida")
    threading.Thread(target=process_after_stop).start()
    return jsonify({"message": "Grabación detenida. Traduciendo..."})

@app.route("/alphabet")
def alphabet():
    """
    Renderiza la página del alfabeto para la traducción.
    """
    return render_template("alphabet.html")

@app.route("/words")
def words():
    """
    Renderiza la página del alfabeto para la traducción.
    """
    return render_template("words.html")

@app.route("/video_feed_alphabet")
def video_feed_alphabet():
    """
    Proporciona la transmisión de video procesada.
    """
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video_feed_words")
def video_feed_words():
    """
    Proporciona la transmisión de video procesada.
    """
    return Response(
        generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/start_tanslate")
def start_translate():
    """
    Redirige a la página del alfabeto para comenzar la traducción.
    """
    return render_template("choice.html")

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

# Ruta para descargar el historial
@app.route('/download_history')
def download_history():
    global historial
    if historial:
        timestamp = datetime.now().strftime("%Y%m%d%H")
        user_history_file = os.path.join(history_folder, f"historial_{timestamp}.txt")
        with open(user_history_file, 'w') as file:
            for palabra in historial:
                file.write(palabra + '\n')
        return send_file(user_history_file, as_attachment=True)
    else:
        return "No hay historial disponible para descargar."

# Definición de rutas
@app.route("/")
def index():
    """
    Renderiza la página de inicio.
    """
    return render_template("index.html")
# Iniciar la aplicación Flask
if __name__ == "__main__":
    app.run("127.0.0.1", 5000, debug=True)
