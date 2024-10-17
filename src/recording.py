from transformers import TimesformerForVideoClassification
from torchvision import transforms
import cv2
import torch
import threading
import queue

# Cargar el modelo
model = torch.load('data/senaliza-final1.pth', map_location=torch.device("cpu"), weights_only=False)
model.eval()

frame_skip = 3
max_frames = 32


palabras = ["Gracias","Hola","Por Favor"]

# Transformación para normalizar los frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_recording(frame_queue):
    frames = []
    frame_count = 0
    print(f"Procesando {frame_queue.qsize()} frames...")

    # Extraer los frames acumulados en la cola
    while not frame_queue.empty():
        if frame_count % frame_skip == 0:
            frame = frame_queue.get()
            frame = transform(frame)
            frames.append(frame)
        frame_count += 1

    # Si hay suficientes frames, hacer predicción
    if len(frames) >= max_frames:
        frames = frames[:max_frames]  # Limitar al máximo permitido
        frames_tensor = torch.stack(frames).unsqueeze(0)
        word_prediction = predict(frames_tensor)
        return palabras[word_prediction]
    else:
        print("No hay suficientes frames para procesar.")

def predict(video):
    with torch.no_grad():
        output = model(video)
        _, predicted = torch.max(output.logits, 1)
        print(f"Predicción: {predicted.item()}")
        return predicted.item()


