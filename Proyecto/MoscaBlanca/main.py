import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog


def aplicar_dilatacion(imagen, kernel):
    dilatada = cv2.dilate(imagen, kernel, iterations=1)
    return dilatada


def dilatacion_nf(image, kernel):
    # Obtener las dimensiones de la imagen y el kernel
    #  [:2] hacemos un slice a lo que devuelve la funcion para solo obtener las dos primeras posiciones
    rows, cols = image.shape[:2]
    k_rows, k_cols = kernel.shape[:2]

    # Calcular el desplazamiento del kernel
    offset_x = k_rows // 2
    offset_y = k_cols // 2

    # Crear una matriz para almacenar la imagen dilatada
    dilated_image = np.zeros_like(image)

    # Recorrer todos los píxeles de la imagen
    for i in range(rows):
        for j in range(cols):
            # Comprobar si el píxel coincide con el kernel
            if image[i, j] == 255:
                # Recorrer todos los elementos del kernel
                for k in range(k_rows):
                    for l in range(k_cols):
                        # Obtener las coordenadas del píxel en la imagen
                        x = i + k - offset_x
                        y = j + l - offset_y

                        # Comprobar si el píxel está dentro de los límites de la imagen
                        if x >= 0 and x < rows and y >= 0 and y < cols:
                            # Dilatar el píxel de la imagen
                            dilated_image[x, y] = 255

    return dilated_image

#def buscar_coincidencias(imagen1, imagen2):
    # Shape devuelve una tupla con las dimensiones de la imagen (alto, ancho, canales), por lo que solo tomamos las dos primeras posiciones
#    height1, width1 = imagen1.shape[:2]
#   height2, width2 = imagen2.shape[:2]
#
#   for i in range(height2 - height1 + 1):
#       for j in range(width2 - width1 + 1):
#           # Obtenemos la region de la imagen2 que coincida con la imagen1 y la comparamos
#            region = imagen2[i:i+height1, j:j+width1]
#            if np.array_equal(region, imagen1):
#                # Si coinciden, pintamos la region de la imagen2 de color gris
#                imagen2[i:i+height1, j:j+width1] = 180
#
#    return imagen2

def buscar_coincidencias(imagenD, imagenR, imagen_origen):
    imagenD_gray = cv2.cvtColor(imagenD, cv2.COLOR_BGR2GRAY)
    imagenR_gray = cv2.cvtColor(imagenR, cv2.COLOR_BGR2GRAY)
    imagen_origen_gray = cv2.cvtColor(imagen_origen, cv2.COLOR_BGR2GRAY)

    resultadoD = cv2.matchTemplate(imagen_origen_gray, imagenD_gray, cv2.TM_CCOEFF_NORMED)
    resultadoR = cv2.matchTemplate(imagen_origen_gray, imagenR_gray, cv2.TM_CCOEFF_NORMED)

    umbral = 0.8

    locD = np.where(resultadoD >= umbral)
    locR = np.where(resultadoR >= umbral)

    for pt in zip(*locD[::-1]):
        cv2.rectangle(imagen_origen, pt, (pt[0] + imagenD_gray.shape[1], pt[1] + imagenD_gray.shape[0]), (255, 0, 0), 2)
        recorte = imagen_origen[pt[1]:pt[1] + imagenD_gray.shape[0], pt[0]:pt[0] + imagenD_gray.shape[1]]
        cv2.imwrite('recorte_D.png', recorte)

    for pt in zip(*locR[::-1]):
        cv2.rectangle(imagen_origen, pt, (pt[0] + imagenR_gray.shape[1], pt[1] + imagenR_gray.shape[0]), (255, 0, 0), 2)
        recorte = imagen_origen[pt[1]:pt[1] + imagenR_gray.shape[0], pt[0]:pt[0] + imagenR_gray.shape[1]]
        cv2.imwrite('recorte_R.png', recorte)

    return imagen_origen

# Cargar las imágenes de las flechas
imagenD = cv2.imread("flechaD.png", cv2.IMREAD_COLOR)
imagenR = cv2.imread("flechaR.png", cv2.IMREAD_COLOR)
# Cargar la imagen original donde se buscarán las coincidencias
imagen_origen = cv2.imread("img.png", cv2.IMREAD_COLOR)

# Buscar y dibujar las coincidencias en la imagen original
imagen_salida = buscar_coincidencias(imagenD, imagenR, imagen_origen.copy())

# Crear una máscara para identificar los píxeles blancos
mascara_blanco = cv2.threshold(imagen_salida, 254, 255, cv2.THRESH_BINARY)[1]

# Iterar sobre los píxeles blancos y asignarles el color azul
for i in range(imagen_salida.shape[0]):
    for j in range(imagen_salida.shape[1]):
        if all(mascara_blanco[i, j]):
            imagen_salida[i, j] = (255, 0, 0)  # Color azul

# Mostrar la imagen de salida con las coincidencias detectadas
cv2.imshow("Coincidencias", imagen_salida)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar la imagen con las coincidencias detectadas
cv2.imwrite("imagen_coincidencias.png", imagen_salida)

def realizar_operacion_or(imagen1_path, imagen2_path, nombre_salida):
    # Cargar las imágenes
    imagen1 = cv2.imread(imagen1_path, cv2.IMREAD_COLOR)
    imagen2 = cv2.imread(imagen2_path, cv2.IMREAD_COLOR)

    # Realizar la operación "or" entre las imágenes
    resultado = cv2.bitwise_or(imagen1, imagen2)

    # Guardar el resultado en un nuevo archivo
    cv2.imwrite(nombre_salida, resultado)

# Ejemplo de uso
imagen1_path = "imagen_coincidencias.png"
imagen2_path = "prueba.png"
nombre_salida = "resultado_or.png"

realizar_operacion_or(imagen1_path, imagen2_path, nombre_salida)

def detectar_lineas(imagen_path, nombre_salida):
    # Cargar la imagen
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    # Aplicar el detector de bordes Canny
    bordes = cv2.Canny(imagen, 50, 150)

    # Aplicar la transformada de Hough para detectar líneas
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, 50, 50, 10)

    # Dibujar las líneas detectadas en la imagen
    imagen_lineas = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)
    if lineas is not None:
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            cv2.line(imagen_lineas, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Líneas azules (BGR)

    # Crear una máscara para identificar los píxeles blancos
    mascara_blanco = cv2.threshold(imagen, 254, 255, cv2.THRESH_BINARY)[1]

    # Convertir la imagen de líneas a color azul en los píxeles blancos
    imagen_lineas[np.where(mascara_blanco)] = (255, 0, 0)

    # Mostrar la imagen con las líneas detectadas
    cv2.imshow("Líneas detectadas", imagen_lineas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar la imagen con las líneas detectadas
    cv2.imwrite(nombre_salida, imagen_lineas)

# Ejemplo de uso
imagen_path_D = "recorte_D.png"
detectar_lineas(imagen_path_D, "lineas_detectadas_D.png")

imagen_path_R = "recorte_R.png"
detectar_lineas(imagen_path_R, "lineas_detectadas_R.png")


def erosion_nf(image, kernel):
    # Obtener las dimensiones de la imagen y el kernel
    rows, cols = image.shape
    k_rows, k_cols = kernel.shape

    # Calcular el desplazamiento del kernel
    offset_x = k_rows // 2
    offset_y = k_cols // 2

    # Crear una matriz para almacenar la imagen erosionada
    eroded_image = np.zeros_like(image)

    # Recorrer todos los píxeles de la imagen
    for i in range(rows):
        for j in range(cols):
            # Comprobar si el píxel coincide con el kernel
            for k in range(k_rows):
                for l in range(k_cols):
                    # Obtener las coordenadas del píxel en la imagen
                    x = i + k - offset_x
                    y = j + l - offset_y

                    # Comprobar si el píxel está dentro de los límites de la imagen
                    if x >= 0 and x < rows and y >= 0 and y < cols:
                        # Comprobar si el píxel en la imagen coincide con el píxel activo del kernel
                        if kernel[k, l] == 1 and image[x, y] == 0:
                            # Erosionar el píxel de la imagen
                            eroded_image[i, j] = 255
                            break
                if eroded_image[i, j] == 255:
                    break

    return eroded_image


def aplicar_erosion(imagen, kernel):
    erosionada = cv2.erode(imagen, kernel, iterations=1)
    return erosionada


def aplicar_hit_or_miss(imagen, kernel):
    transformacion = cv2.morphologyEx(imagen, cv2.MORPH_HITMISS, kernel)
    return transformacion


def aplicar_canny(imagen, umbral_min, umbral_max):
    bordes = cv2.Canny(imagen, umbral_min, umbral_max)
    return bordes


def segmentar_flechas(imagen_binaria):
    # Definir el kernel
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    # Aplicar dilatación para resaltar las características de las flechas
    dilatada = aplicar_dilatacion(imagen_binaria, kernel)

    # Aplicar erosión para eliminar ruido
    erosionada = aplicar_erosion(dilatada, kernel)

    # Definir el kernel para Hit or Miss
    kernel_hitmiss = np.array([[0, 1, 0],
    
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.uint8)

    # Aplicar Hit or Miss para identificar las flechas
    flechas = aplicar_hit_or_miss(erosionada, kernel_hitmiss)

    # Aplicar Canny para detectar bordes
    umbral_min = 100
    umbral_max = 200
    bordes = aplicar_canny(erosionada, umbral_min, umbral_max)

    # Mostrar las imágenes resultantes
    cv2.imshow('Dilatada', dilatada)
    cv2.imshow('Erosionada', erosionada)
    cv2.imshow('Flechas', flechas)
    cv2.imshow('Bordes (Canny)', bordes)
    # Guardar las imágenes resultantes
    cv2.imwrite('dilatada.png', dilatada)
    cv2.imwrite('erosionada.png', erosionada)
    cv2.imwrite('flechas.png', flechas)
    cv2.imwrite('bordes.png', bordes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def printArray(array, title):
    plt.imshow(array, cmap='gray')
    plt.axis('off')  # Opcional: Ocultar los ejes
    # Titulo de la imagen
    plt.title(title)
    plt.show()

# Cargar la imagen binarizada de las flechas


kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)
# Cargando la imagen
# filepath = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=(
    # ("Imagenes JPG", "*.jpg"), ("Imagenes PNG", "*.png"), ("Imagenes BMP", "*.bmp"), ("Todos los archivos", "*.*")))
# Obteniendo imagen binaria
# imagen_binaria = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
# Segmentacion de flechas
# segmentar_flechas(imagen_binaria)
np.set_printoptions(threshold=np.inf)
# Mostrando imagen original
# cv2.imshow("Original", imagen_binaria)
# Aplicando dilatacion
# creando copia de la imagen y pasando a las funciones correspondientes
# resultado_dilatacion = dilatacion_nf(imagen_binaria.copy(), kernel)
# resultado_erosion = erosion_nf(imagen_binaria.copy(), kernel)
# Mostrando resultados de dilatacion y erosion
# printArray(resultado_dilatacion, "Dilatacion")
# printArray(resultado_erosion, "Erosion")

# Coincidencias

# La primer imagen es la que sera buscada sobre la segunda
#resultado2 = buscar_coincidencias(cv2.imread("flechaD.png"),cv2.imread("img.png"))
#printArray(resultado2, "Coincidencias")

# La segunda imagen es la que sera buscada
#resultado3 = buscar_coincidencias(cv2.imread("flechaR.png"),cv2.imread("img.png"))
#printArray(resultado3, "Coincidencias")

# Segmentar las flechas
