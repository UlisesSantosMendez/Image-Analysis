import cv2
import numpy as np

def buscar_coincidencias(imagen1, imagen2, margen_error=3):
    # Obtener dimensiones de las imágenes
    # Shape devuelve una tupla con las dimensiones de la imagen (alto, ancho, canales)
#  [:2] hacemos un slice a lo que devuelve la funcion para solo obtener las dos primeras posiciones

    height1, width1 = imagen1.shape[:2]
    height2, width2 = imagen2.shape[:2]
    coincidencias = []
    # Recorrer los píxeles de la segunda imagen para buscar coincidencias
    for y in range(height2 - height1 + 1):
        for x in range(width2 - width1 + 1):
            # Extraer la región de la segunda imagen del mismo tamaño que la primera imagen
            region = imagen2[y:y+height1, x:x+width1]

            # Verificar si la región coincide con la imagen 1 (con margen de error)
            if np.sum(np.abs(region - imagen1)) <= margen_error * 255:
                # Coincidencia encontrada
                coincidencias.append((x, y))

    # Crear una matriz para almacenar las coincidencias
    matriz_coincidencias = np.zeros_like(imagen2, dtype=np.uint8)

    # Rellenar la matriz con las coincidencias encontradas
    for x, y in coincidencias:
        matriz_coincidencias[y:y+height1, x:x+width1] = imagen1

    return matriz_coincidencias

# Cargar las dos imágenes
imagen1 = cv2.imread('flecha1.jpg', cv2.IMREAD_GRAYSCALE)
imagen2 = cv2.imread('rvb.jpg', cv2.IMREAD_GRAYSCALE)

# Convertir los valores a 0 y 255
_, imagen1 = cv2.threshold(imagen1, 127, 255, cv2.THRESH_BINARY)
_, imagen2 = cv2.threshold(imagen2, 127, 255, cv2.THRESH_BINARY)

# Buscar coincidencias
matriz_coincidencias = buscar_coincidencias(imagen1, imagen2)

# Mostrar la matriz de coincidencias
cv2.imshow("Coincidencias", matriz_coincidencias)
cv2.waitKey(0)
cv2.destroyAllWindows()
