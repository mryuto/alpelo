import os

import cv2
import mediapipe as mp

# -----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGES_PATHS = {
    "VICTORIA": os.path.join(BASE_DIR, "img", "victoria.jpeg"),
    "PULGAR": os.path.join(BASE_DIR, "img", "pulgar.jpeg"),
    "MANO_ABIERTA": os.path.join(BASE_DIR, "img", "mano_abierta.jpeg"),
    "PUÑO": os.path.join(BASE_DIR, "img", "punho.jpeg"),
    "UN_DEDO": os.path.join(BASE_DIR, "img", "undedo.jpg"),
}

# Carga las imágenes con canal alfa para permitir transparencia si existe.
IMAGENES = {
    nombre: cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    for nombre, ruta in IMAGES_PATHS.items()
}

# Inicialización de MediaPipe para detección de manos.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)


# -----------------------------------------------------------------------------
# LÓGICA DE GESTOS
# -----------------------------------------------------------------------------
def dedos_levantados(hand):
    """
    Devuelve el estado de cada dedo en este orden:
    [pulgar, índice, medio, anular, meñique]
    True = dedo levantado, False = dedo cerrado.
    """
    dedos = []

    # Índice, medio, anular y meñique:
    # Comparamos la punta del dedo con la articulación de la falange anterior.
    puntos = [
        (8, 6),   # índice
        (12, 10), # medio
        (16, 14), # anular
        (20, 18), # meñique
    ]

    for punta, articulacion in puntos:
        dedos.append(hand.landmark[punta].y < hand.landmark[articulacion].y)

    # Pulgar: se considera levantado si está más a la derecha que la falange.
    dedos.insert(0, hand.landmark[4].x < hand.landmark[3].x)

    return dedos


def reconocer_gesto(dedos):
    """Devuelve el nombre del gesto a partir del estado de los dedos."""
    if dedos == [False, False, False, False, False]:
        return "PUÑO"

    if dedos == [True, True, True, True, True]:
        return "MANO ABIERTA"

    if dedos == [False, True, True, False, False]:
        return "VICTORIA"

    if dedos == [False, True, False, False, False]:
        return "UN DEDO"

    if dedos == [True, False, False, False, False]:
        return "PULGAR"

    return "DESCONOCIDO"


# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES DE VISUALIZACIÓN
# -----------------------------------------------------------------------------
def superponer_imagen(frame, imagen, escala=1.0):
    """Inserta una imagen en la esquina inferior derecha del frame."""
    if imagen is None:
        return frame

    alto, ancho = imagen.shape[:2]
    nuevo_ancho = int(ancho * escala)
    nuevo_alto = int(alto * escala)

    imagen_redimensionada = cv2.resize(
        imagen,
        (nuevo_ancho, nuevo_alto),
        interpolation=cv2.INTER_AREA,
    )

    x0 = frame.shape[1] - nuevo_ancho - 20
    y0 = frame.shape[0] - nuevo_alto - 20
    x1 = x0 + nuevo_ancho
    y1 = y0 + nuevo_alto

    # Si la imagen tiene canal alfa, se conserva la transparencia.
    if imagen_redimensionada.shape[2] == 4:
        overlay = imagen_redimensionada[:, :, :3]
        mask = imagen_redimensionada[:, :, 3]
        roi = frame[y0:y1, x0:x1]
        roi_mask = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
        overlay_mask = cv2.bitwise_and(overlay, overlay, mask=mask)
        frame[y0:y1, x0:x1] = cv2.add(roi_mask, overlay_mask)
    else:
        frame[y0:y1, x0:x1] = imagen_redimensionada

    return frame


def procesar_gesto(frame, gesto):
    """Muestra la imagen asociada al gesto detectado."""
    gestos = {
        "VICTORIA": (IMAGENES["VICTORIA"], 0.50),
        "PULGAR": (IMAGENES["PULGAR"], 0.70),
        "MANO ABIERTA": (IMAGENES["MANO_ABIERTA"], 0.50),
        "PUÑO": (IMAGENES["PUÑO"], 0.50),
        "UN DEDO": (IMAGENES["UN_DEDO"], 1.00),
    }

    if gesto in gestos:
        imagen, escala = gestos[gesto]
        frame = superponer_imagen(frame, imagen, escala)

    return frame


# -----------------------------------------------------------------------------
# BUCLE PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear la imagen como espejo para una experiencia más natural.
        frame = cv2.flip(frame, 1)

        # MediaPipe trabaja con RGB, mientras OpenCV usa BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)

        gesto = "DESCONOCIDO"

        if resultado.multi_hand_landmarks:
            for hand in resultado.multi_hand_landmarks:
                # Indicaciones de la mano sobre el frame.
                mp_drawing.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                )

                dedos = dedos_levantados(hand)
                gesto = reconocer_gesto(dedos)

                cv2.putText(
                    frame,
                    gesto,
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                )

        frame = procesar_gesto(frame, gesto)

        cv2.imshow("Reconocimiento de gestos", frame)

        # Presiona ESC para cerrar la aplicación.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()