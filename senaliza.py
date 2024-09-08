from flask import Flask
from flask import render_template
from flask import Response
import cv2
from src.senaliza_v2 import *
from src.camara_flask import initialize_camera, process_frame, model, hands, letterpred

app = Flask(__name__)
# Inicializar cámara
cap = initialize_camera()

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Procesar frame y hacer predicción
            frame = process_frame(frame, model, hands, letterpred)

            # Codificar la imagen como JPEG para enviar al cliente
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


#Rutas - (argumentos: Url) - Función
@app.route('/') 
def inicio():
    return render_template('index.html')

@app.route("/video_feed")
def video_feed():
     return Response(generate_frames(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")

#Iniciar app
if __name__ == '__main__':
    app.run('127.0.0.1', 5000, debug=False)