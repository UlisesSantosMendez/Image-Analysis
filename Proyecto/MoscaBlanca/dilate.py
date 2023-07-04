import numpy as np
import cv2
def dilatacion_nf(imagen):
    height, width = imagen.shape
    printArray(np.array(imagen), "imagen original")
    result = np.zeros((height, width), dtype=np.uint8)

    # Definir el kernel de dilatación
    kernel = np.array([[1, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    # Iterar sobre cada píxel de la imagen
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Verificar si el píxel central coincide con el kernel
            if imagen[i, j] == 255:
                result[i-1:i+2, j-1:j+2] = kernel
                result[i, j] = 1
                imagen[i-1:i+2, j-1:j+2] = kernel

    printArray(np.array(imagen), "imagen Dilatada")
    result = result.astype(np.uint8)
    return result


def erosion_nf(imagen):
    height, width = imagen.shape
    printArray(np.array(imagen),"Erosion original")

    result = np.zeros((height, width), dtype=np.uint8)

    # Definir el kernel de dilatación
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (imagen[i, j] == 255 and
                imagen[i-1, j] == 255 and
                imagen[i+1, j] == 255 and
                imagen[i, j-1] == 255 and
                imagen[i, j+1] == 255):
                result[i, j] = 255

    return result



imagen_binaria = cv2.imread("flecha1.jpg", 0)
# imagen_binaria = cv2.threshold(imagen_binaria, 127, 255, cv2.THRESH_BINARY)[1]
kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)
cv2.imshow("Original", imagen_binaria)
dilatada = dilate(imagen_binaria, kernel)
erosionada = erode(imagen_binaria, kernel)
cv2.imshow("Dilatada", dilatada)
cv2.imshow("Erosionada", erosionada)
cv2.waitKey(0)
cv2.destroyAllWindows()
