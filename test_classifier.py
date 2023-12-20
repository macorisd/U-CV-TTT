###########################################################################################

# Archivo destinado a probar el clasificador de circulos y cruces.
# Procedimiento:
    # Tener los archivos 'class_x.txt' y 'class_o.txt' con los resultados del entrenamiento del clasificador.
    # Llenar carpetas 'Class_X_test' y 'Class_O_test' con las imagenes de prueba (distintas a las de entrenamiento).
    # Ejecutar este archivo.
    # Se mostraran las probabilidades de pertenencia a cada clase de cada imagen de prueba.

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

    # Calcular la matriz de transformación de perspectiva
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

# Calcula la probabilidad de que 'img' pertenezca a cada clase.
# devuelve un array de 2 posiciones con [prob(X), prob(O)]
def class_probability_xo(img, votes, dark_threshold):
    probability = [None, None]

    for iter in range(2):        
        assert img.shape[0] * img.shape[1] == votes[iter].size

        pixels = img.shape[0]

        prob_accumulated = 0

        for i in range(pixels):
            for j in range(pixels):
                if img[i, j] < dark_threshold:
                    prob_accumulated += votes[iter][i][j]
                
        probability[iter] = prob_accumulated

    return probability

def main():
    script_directory = os.path.dirname(os.path.abspath(__file__)).replace('\\', '\\\\')

    pixels = 16
    dark_threshold = 250    

    # Rutas de las carpetas de entrada con las imagenes de prueba
    dir_class_x_test = script_directory + "\\Classifier\\Class_X_test"
    dir_class_o_test = script_directory + "\\Classifier\\Class_O_test"

    dirs_list = [dir_class_x_test, dir_class_o_test]

    votes_xo = np.zeros((2, pixels, pixels), dtype=int)    

    file_path = ''
    symbol = ''

    for iter in range(2):
        if (iter == 0):
            file_path = script_directory + "\\Classifier\\class_x.txt"            
        else:
            file_path = script_directory + "\\Classifier\\class_o.txt"

        rows_list = []

        # Leer el archivo y procesar cada linea
        with open(file_path, "r") as file:
            for line in file:                
                values = line.strip().split(",")                
                row = list(map(int, values))
                rows_list.append(row)

        # Convertir la lista de filas a un array bidimensional
        votes_xo[iter] = np.array(rows_list)

        #print(f"Votos {symbol}:\n{votes_xo[iter]}")

    max_probs = [0, 0]

    for i in range(pixels):
        for j in range(pixels):
            max_probs[0] += votes_xo[0][i][j]
            max_probs[1] += votes_xo[1][i][j]

    max_prob_class = 0

    if (max_probs[1] > max_probs[0]):
        max_prob_class = 1

    for iter in range(2): # [iter1: X] [iter2: O]  
        if (iter == 0):            
            symbol = 'X'            
        else:            
            symbol = 'O'            

        dir = dirs_list[iter]         

        # Lista de archivos en la carpeta de entrada
        files = os.listdir(dir)
        
        print(f"\n\n\nPROBABILIDADES CLASE {symbol}\n")

        # Iterar sobre cada archivo en la carpeta de entrada
        for file in files:
            # Leer la imagen
            image = cv2.imread(os.path.join(dir, file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            print(f"Len: {len(files)}")

            # Convertir imagen a cuadrado
            image = to_square(image)

            # Pixelar imagen
            image = pixelate(image, pixels)

            prob = class_probability_xo(image, votes_xo, dark_threshold)

            # maximo del menor -------- mi valor
            # maximo del mayor -------- x

            prob[1-max_prob_class] = int(prob[1-max_prob_class] * max_probs[max_prob_class] / max_probs[1-max_prob_class])

            class_result = "EMPATE"
            
            if prob[0] > prob[1]:
                class_result = "X"

            elif prob[1] > prob[0]:
                class_result = "O"

            print(f"Probabilidad de X: {prob[0]}\t\tProbabilidad de O: {prob[1]}\t\tClasificado como: {class_result}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()