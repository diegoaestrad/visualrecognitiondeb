import cv2
import base64
import requests

# 1. Configuración de la API de NVIDIA NIM
API_KEY = "TU_NVAPI_KEY_AQUÍ"
API_URL = "https://nvidia.com"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 2. Inicializar la cámara web (0 es la cámara por defecto)
cap = cv2.VideoCapture(0)

print("Presiona 'Espacio' para analizar la imagen o 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mostrar el flujo de la cámara en vivo
    cv2.imshow("Reconocimiento Visual - OpenCode", frame)
    key = cv2.waitKey(1) & 0xFF

    # Si el usuario presiona Espacio, se procesa el fotograma
    if key == ord(' '):
        print("Analizando fotograma...")
        
        # Codificar la imagen actual a formato JPEG y luego a Base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Carga de datos para el modelo multimodal Nemotron
        payload = {
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe con alta precisión qué objeto o acción ves en esta imagen."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            "max_tokens": 300
        }

        # Enviar a NVIDIA NIM
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            resultado = response.json()['choices'][0]['message']['content']
            print(f"\n[Resultado de la IA]: {resultado}\n")
        else:
            print(f"Error en la API: {response.text}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
