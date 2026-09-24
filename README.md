# Alpelo 👍

Programa de reconocimiento de gestos con la mano usando la cámara web, OpenCV y MediaPipe.

## Requisitos

- Python 3.10 o superior
- Una cámara web
- Las dependencias del proyecto:

```bash
pip install opencv-python mediapipe
```

## Ejecución

Desde la carpeta del proyecto, ejecuta:

```bash
python alpelo.py
```

Se abrirá una ventana con la imagen de la cámara. Para cerrar el programa, pulsa `Esc`.

## Gestos reconocidos

- Puño
- Mano abierta
- Victoria
- Un dedo
- Pulgar

Al detectar un gesto, el programa muestra su nombre y la imagen asociada en la esquina inferior derecha.

## Estructura

```text
alpelo.py
img/
  mano_abierta.jpeg
  pulgar.jpeg
  punho.jpeg
  undedo.jpg
  victoria.jpeg
```
