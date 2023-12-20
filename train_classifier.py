###########################################################################################

# Archivo destinado a entrenar el clasificador de circulos y cruces.
# Procedimiento:
    # Llenar carpetas 'Class_X' y 'Class_O' con las imagenes de entrenamiento.
    # Ejecutar este archivo.
    # Se añadiran a 'Class_X_squared' y 'Class_O_squared' las imagenes cuadradas.
    # Se añadiran a 'Class_X_pixelated' y 'Class_O_pixelated' las imagenes pixeladas.
    # Se añadiran a 'class_x.txt' y 'class_o.txt' los resultados numericos del clasificador de cada clase.

###########################################################################################

import os
import cv2
import numpy as np

# Convierte una imagen 'img' a cuadrado
# devuelve la imagen 'img' convertida a cuadrado
def to_square(img):

    height = img.shape[0]
    width = img.shape[1]

    # Definir los puntos de entrada y salida
    pts1 = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]])
    pts2 = np.float32([[0, 0], [0, 500], [500, 500], [500, 0]])

    # Crear una mascara para la region de interes
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(mask, [pts1.astype(int)], (255, 255, 255))

    # Calcular la matriz de transformacion de perspectiva
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # Aplicar la transformacion a 'img'
    square_img = cv2.warpPerspective(img, M, (500, 500))

    return square_img

# Redimensiona 'img' a (n x n) dimensiones
# devuelve la imagen 'img' redimensionada
def pixelate(img, n):
    # Redimensionar la imagen a (n x n) dimensiones
    pixelated_img = cv2.resize(img, (n, n), interpolation=cv2.INTER_LINEAR)     

    # Escalar la imagen a las dimensiones originales
    #pixelated_img = cv2.resize(pixelated_img, (500, 500), interpolation=cv2.INTER_NEAREST)    

    return pixelated_img

def main():
    script_directory = os.path.dirname(os.path.abspath(__file__)).replace('\\', '\\\\')

    pixels = 16
    dark_threshold = 250

    guardar_imagenes = True

    # Ruta de la carpeta de entrada y salida
    dir_class_x = script_directory + "\\Classifier\\Class_X"
    dir_class_o = script_directory + "\\Classifier\\Class_O"
    dir_class_x_square = script_directory + "\\Classifier\\Class_X_square"
    dir_class_o_square = script_directory + "\\Classifier\\Class_O_square"
    dir_class_x_pixelated = script_directory + "\\Classifier\\Class_X_pixelated"
    dir_class_o_pixelated = script_directory + "\\Classifier\\Class_O_pixelated"

    dirs_list = [[dir_class_x, dir_class_x_square, dir_class_x_pixelated], [dir_class_o, dir_class_o_square, dir_class_o_pixelated]]

    votes_xo = np.zeros((2, pixels, pixels), dtype=int)

    file_path = ''
    symbol = ''

    for iter in range(2): # [iter1: X] [iter2: O]        
        dirs = dirs_list[iter] 
        votes = votes_xo[iter]

        # Lista de archivos en la carpeta de entrada
        files = os.listdir(dirs[0])

        cont = 0

        if (iter == 0):
            file_path = script_directory + "\\Classifier\\class_x.txt"
            symbol = 'X'
        else:
            file_path = script_directory + "\\Classifier\\class_o.txt"
            symbol = 'O'


        # Iterar sobre cada archivo en la carpeta de entrada
        for file in files:
            # Leer la imagen
            image = cv2.imread(os.path.join(dirs[0], file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Convertir imagen a cuadrado
            image = to_square(image)

            # Guardar la imagen cuadrada en la carpeta de salida
            if guardar_imagenes:
                cv2.imwrite(os.path.join(dirs[1], f"{file}"), image)   

            # Pixelar imagen
            image = pixelate(image, pixels)            

            this_votes = image < dark_threshold

            cont += 1

            for i in range(pixels):
                for j in range(pixels):
                    if this_votes[i][j]:
                        votes[i][j] += 1

            # Guardar la imagen pixelada en la carpeta de salida
            if guardar_imagenes:
                cv2.imwrite(os.path.join(dirs[2], f"{file}"), image)  
        
        print(f"Imagenes [{symbol}] convertidas a cuadrado y pixeladas.")
        print(f"Valor maximo: {str(cont)}      Votos {symbol}:\n{str(votes)}") 
               
        votes_min = np.min(votes)
        votes_max = np.max(votes)
        range_votes = votes_max - votes_min

        # Aplicar la formula de normalizacion
        # normalized_votes = (votes / cont) * 255
        normalized_votes = ((votes - votes_min) / range_votes) * 255

        # Asegurarse de que los valores son enteros en el rango [0, 255]
        normalized_votes = np.clip(normalized_votes, 0, 255)    
        normalized_votes = normalized_votes.astype(int)
        
        print(f"Valor maximo {symbol} normalizado: {str(np.max(normalized_votes))}")
        print(f"Votos {symbol} normalizados:\n {str(normalized_votes)}")

        # Escribir en fichero
        with open(file_path, "w") as file:
            # Iterar sobre las filas del array
            for row in normalized_votes:
                # Convertir cada elemento de la fila a cadena y unirlos con comas
                row_str = ",".join(map(str, row))                
                file.write(row_str + "\n")
        
        print(f"El array normalizado se ha escrito en el archivo: {file_path}")
        
        votes_display = normalized_votes.astype(np.uint8)

        zoom_factor = 30
        votes_zoomed = cv2.resize(votes_display, (votes_display.shape[1] * zoom_factor, votes_display.shape[0] * zoom_factor), interpolation=cv2.INTER_NEAREST)

        # Mostrar la imagen resultante
        cv2.imshow("class " + str(iter), votes_zoomed)
        cv2.waitKey(1)

    print("Proceso completado.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()