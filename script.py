import os
from moviepy.editor import VideoFileClip

def convert_videos_to_gifs(input_folder, output_folder, duration=None):
    # Asegura que la carpeta de salida exista
    os.makedirs(output_folder, exist_ok=True)

    # Recorre todos los archivos de la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith((".mp4", ".mov", ".avi", ".mkv")):  # Añade más extensiones si es necesario
            video_path = os.path.join(input_folder, filename)
            gif_name = os.path.splitext(filename)[0] + ".gif"
            gif_path = os.path.join(output_folder, gif_name)

            try:
                print(f"Convirtiendo {filename} a GIF...")
                with VideoFileClip(video_path) as video:
                    # Si se especifica una duración, se recorta el video
                    if duration:
                        video = video.subclip(0, duration)

                    # Guarda el GIF con la mejor relación calidad-tamaño
                    video.write_gif(gif_path, program='ffmpeg')
                print(f"{filename} convertido exitosamente a {gif_name}")
            except Exception as e:
                print(f"Error al convertir {filename}: {e}")

# Rutas relativas de las carpetas de entrada y salida
input_folder = "./Videos"
output_folder = "./Viejo"

# Llamada a la función
convert_videos_to_gifs(input_folder, output_folder, duration=5)  # Cambia la duración si es necesario


