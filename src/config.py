import os

# Cargar la clave API desde una variable de entorno
API_KEY = os.getenv('API_FOOTBALL_KEY', "0fd247b049e29f77d89dce2eea2d08f1")

if not API_KEY:
    raise ValueError("La variable de entorno 'API_FOOTBALL_KEY' no está establecida.")
