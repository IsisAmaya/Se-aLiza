import pytest
import json

@pytest.fixture
def client():
    from senaliza import app 
    with app.test_client() as client:
        yield client

# HU 005: Historial de traducciones
def test_historial_de_traducciones(client):
    # Inicia la grabación
    response = client.post('/start_recording')
    assert response.status_code == 200
    data = json.loads(response.get_data(as_text=True))
    assert data['message'] == "Grabación iniciada"

    # Detiene la grabación
    response = client.post('/stop_recording')
    assert response.status_code == 200
    data = json.loads(response.get_data(as_text=True))
    assert "No hay datos en la cola para procesar." in data['message']
    
    # Obtén el historial
    response = client.get('/download_history')
    assert response.status_code == 200
    assert "No hay historial disponible para descargar." in response.get_data(as_text=True)

def test_historial_vacio(client):
    response = client.get('/download_history')
    assert response.status_code == 200
    assert "No hay historial disponible para descargar." in response.get_data(as_text=True)

# HU 006: Traducción con uso de texto a voz
def test_traduccion_texto_a_voz(client):
    # Simulamos datos en la cola antes de iniciar
    from senaliza import frame_queue
    frame_queue.put("dummy_frame")  # Agregamos un frame de prueba
    
    response = client.post('/start_recording')
    assert response.status_code == 200
    
    response = client.post('/stop_recording')
    assert response.status_code == 200
    data = json.loads(response.get_data(as_text=True))
    assert "Traduciendo..." in data['message']

def test_traduccion_sin_datos(client):
    # Aseguramos que la cola esté vacía
    from senaliza import frame_queue
    while not frame_queue.empty():
        frame_queue.get()
        
    response = client.post('/stop_recording')
    assert response.status_code == 200
    data = json.loads(response.get_data(as_text=True))
    assert "No hay datos en la cola para procesar." in data['message']

# HU 004: Accesibilidad de texto
def test_accesibilidad_texto(client):
    response = client.get('/alphabet')
    assert response.status_code == 200
    assert "Alfabeto en Lengua de Señas" in response.get_data(as_text=True)

def test_accesibilidad_texto_no_encontrado(client):
    response = client.get('/alphabet')
    assert response.status_code == 200
    assert "texto no existente" not in response.get_data(as_text=True)

# HU NF 009: Navegabilidad de la interfaz
def test_navegabilidad(client):
    response = client.get('/')
    assert response.status_code == 200
    assert "Biblioteca" in response.get_data(as_text=True)

def test_navegacion_a_pagina_inexistente(client):
    response = client.get('/pagina_inexistente')
    assert response.status_code == 404

# HU 012: Material educativo
def test_material_educativo(client):
    response = client.get('/biblioteca')
    assert response.status_code == 200
    assert "Biblioteca de Señas" in response.get_data(as_text=True)

def test_material_educativo_sin_recursos(client):
    response = client.get('/biblioteca')  # Supón que no hay recursos
    assert response.status_code == 200
    assert "No hay recursos disponibles" not in response.get_data(as_text=True)
