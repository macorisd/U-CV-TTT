###########################################################################################

# Proyecto 'Ultimate Computer Vision TicTacToe'
# Objetivo: simular un jugador de tres en raya, cuyos ojos son la camara de un dispositivo movil.
# Inicio: 29/11/2023

# Instalacion:
    # Seguir la guia de instalacion detallada en 'install.txt'.    

# Modos de juego: Facil, Medio, Pesadilla.

# Procedimiento:
    # Conectar el dispositivo movil y el ordenador a la misma red Wi-Fi (o conectarse desde el ordenador a los datos moviles del telefono).
    # Abrir DroidCam en el dispositivo movil.
    # Ejecutar este archivo, o el archivo 'U-CV-TTT.pyw'
    # Hacer clic en el boton 'IP/Port'
    # Introducir la IP y el puerto proporcionados por DroidCam en la ventana de 'IP/Port'.
    # Seleccionar un modo de juego.
    # Capturar imagenes con la barra espaciadora. Salir con 'q'.

# Tips:
    # Intentar usar folios blancos y rotulador negro.
    # Cuadrar el tablero de tres en raya aproximadamente en la cuadricula proporcionada, pero no acercarlo demasiado.
    # Puede dibujar o no dibujar los movimientos del bot en el papel. Es indiferente.

# Avanzado:
    # Puede entrenar su propio clasificador de simbolos con el archivo 'train_classifier.py'.
    # Puede probar el clasificador de simbolos con el archivo 'test_classifier.py'.

# Made with <3 by Macoris Decena Gimenez
# https://github.com/macorisd

# Para reportar bugs y errores: macorisd@gmail.com

###########################################################################################

import math
import os
import cv2
import numpy as np
from datetime import datetime
from shapely import LineString
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import messagebox, PhotoImage
import time
from PIL import Image


######################### FUNCIONES DE LAS PRACTICAS


# Suavizado Guassiano 
# (Chapter 5.1)
def gaussian_smoothing(image, sigma, w_kernel):
    """ Blur and normalize input image.   
    
        Args:
            image: Input image to be binarized
            sigma: Standard deviation of the Gaussian distribution
            w_kernel: Kernel aperture size
                    
        Returns: 
            binarized: Blurred image
    """   
    
    # Define 1D kernel
    s=sigma
    w=w_kernel
    kernel_1D = [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-w,w+1)]
    
    # Apply distributive property of convolution
    kernel_2D = np.outer(kernel_1D,kernel_1D)
    
    # Blur image
    smoothed_img = cv2.filter2D(image,cv2.CV_8U,kernel_2D)
    
    # Normalize to [0 254] values
    smoothed_norm = np.array(image.shape)
    smoothed_norm = cv2.normalize(smoothed_img,None, 0, 255, cv2.NORM_MINMAX)
    
    return smoothed_norm


######################### FUNCIONES PROPIAS


# Mostrar las figuras pasadas como parametro en la misma ventana
def show_window(e1, e2):
    title = "U-CV-TTT"
    
    if gamemode == "easy":
        title = "U-CV-TTT: Easy mode   |   presiona 'q' para salir"
    elif gamemode == "intermediate":
        title = "U-CV-TTT: Intermediate mode   |   presiona 'q' para salir"
    elif gamemode == "nightmare":
        title = "U-CV-TTT: Nightmare mode   |   presiona 'q' para salir"

    cv2.imshow(title, np.hstack((e1, e2)))

# Dibujar cuadrado con margenes 'd_v' y 'd_h' para encajar el tablero
# devuelve la imagen 'img' recortada segun ese cuadrado
def add_square(img, d_v, d_h):
    height, width, _ = img.shape 
    points = np.array([
        [(d_h, d_v), (d_h, height - d_v)],                  # Vertical izquierda
        [(width - d_h, d_v), (width - d_h, height - d_v)],  # Vertical derecha
        [(d_h, d_v), (width - d_h, d_v)],                   # Horizontal abajo
        [(d_h, height - d_v), (width - d_h, height - d_v)]  # Horizontal arriba
    ], dtype=np.int32)

    for line in points:
        cv2.line(img, line[0], line[1], (255, 183, 194), 3, cv2.LINE_AA)
    
    return img[d_v:height-d_v, d_h:width-d_h]

# Encontrar intersecciones en el array de lineas 'lines'
# devuelve un array de intersecciones (puntos)
def find_intersections(lines):
    intersections = []
    
    # Convertir las lineas a objetos LineString de shapely
    line_objects = [LineString([line[0], line[1]]) for line in lines]
    
    # Encontrar intersecciones
    for i in range(len(line_objects)):
        for j in range(i + 1, len(line_objects)):
            intersection = line_objects[i].intersection(line_objects[j])
                        
            if intersection.is_empty:
                continue
                        
            if intersection.geom_type == 'Point':
                intersections.append((int(intersection.x), int(intersection.y)))

    if len(intersections) == 0:
        return None

    return intersections

# Dibujar cuadricula en una celda del tablero
# devuelve la imagen 'img' de la celda con la cuadricula dibujada
def draw_cell(img, intersections, p1, p2, p3, p4):
    # Convertir las tuplas a enteros
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    p3 = (int(p3[0]), int(p3[1]))
    p4 = (int(p4[0]), int(p4[1]))
    
    # Crear una mascara vacia del mismo tamaño que la imagen
    mask = np.zeros_like(img)

    # Definir los vertices del pologono que representa la cuadricula
    pts = np.array([intersections[p1[0]][p1[1]], intersections[p2[0]][p2[1]],
                    intersections[p3[0]][p3[1]], intersections[p4[0]][p4[1]]], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Dibujar el poligono en la mascara
    cv2.fillPoly(mask, [pts], color=(255, 255, 255))

    # Fusionar la mascara con la imagen original usando la operacion OR
    result = cv2.bitwise_or(img, mask)

    cv2.line(result, intersections[p1[0]][p1[1]], intersections[p2[0]][p2[1]], (149, 149, 255), 3, cv2.LINE_AA)
    cv2.line(result, intersections[p2[0]][p2[1]], intersections[p3[0]][p3[1]], (149, 149, 255), 3, cv2.LINE_AA)
    cv2.line(result, intersections[p3[0]][p3[1]], intersections[p4[0]][p4[1]], (149, 149, 255), 3, cv2.LINE_AA)
    cv2.line(result, intersections[p4[0]][p4[1]], intersections[p1[0]][p1[1]], (149, 149, 255), 3, cv2.LINE_AA)

    return result

# Dibujar el simbolo 'symbol' en la celda dada por 'intersections' y los 4 puntos 'pi'
# devuelve la imagen 'img' con el simbolo dibujado
def draw_xo_cell(img, symbol, highlighted, intersections, p1, p2, p3, p4):
    global image_x_highlighted, image_x, image_o_highlighted, image_o

    if symbol == 'x':
        if highlighted:
            symbol_img = image_x_highlighted
        else:
            symbol_img = image_x
    elif symbol == 'o':
        if highlighted:
            symbol_img = image_o_highlighted
        else:
            symbol_img = image_o

    p1 = intersections[p1[0]][p1[1]]
    p2 = intersections[p2[0]][p2[1]]
    p3 = intersections[p3[0]][p3[1]]
    p4 = intersections[p4[0]][p4[1]]

    line1 = LineString([p1, p3])
    line2 = LineString([p2, p4])

    intersection = line1.intersection(line2)

    # Comprobar si hay una interseccion valida
    if intersection.is_empty or intersection.geom_type != 'Point':
        return None

    return overlay_images(img, symbol_img, (int(intersection.x), int(intersection.y)))

def overlay_images(background_img, symbol_img, position):
    # Convertir las imagenes a objetos Pillow
    background_img = Image.fromarray(background_img)

    # Garantizar que la imagen del simbolo tenga un canal alfa (transparencia)
    if symbol_img.mode != 'RGBA':
        symbol_img = symbol_img.convert('RGBA')

    # Redimensionar la imagen del simbolo    
    symbol_img = symbol_img.resize(tuple(int(dim * 1/40) for dim in symbol_img.size), Image.LANCZOS)

    # Calcular la posicion para que el centro de la imagen superpuesta coincida con la posicion especificada
    new_position = (position[0] - symbol_img.width // 2, position[1] - symbol_img.height // 2)

    # Crear y superponer imagen con fondo transparente
    symbol_img_transp = Image.new('RGBA', background_img.size, (0, 0, 0, 0))
    symbol_img_transp.paste(symbol_img, new_position, symbol_img)    
    background_img.paste(symbol_img_transp, (0, 0), symbol_img_transp)

    # Convertir la imagen de Pillow a Numpy
    result = np.array(background_img)

    return result

# Recortar celda para desechar informacion irrelevante
# devuelve la imagen recortada
def crop_cell(img, iter):
    cropped = np.copy(img)

    cell_h, cell_w, _ = cropped.shape  
    
    # Recortar el 5% por cada lado para desechar posibles bordes    
    cropped = cropped[int(cell_h*0.05):int(cell_h-cell_h*0.05),int(cell_w*0.05):int(cell_w-cell_w*0.05)]
    cell_h, cell_w, _ = cropped.shape

    while count_black(cropped[:,0]) > 0.7*(cropped[:,0].shape[0] * cropped[:,0].shape[1]): # Recortar 5% izquierda
        cropped = cropped[:,int(cell_w*0.05):]
        cell_w = cropped.shape[1]

    while count_black(cropped[:,cell_w-1]) > 0.7*(cropped[:,cell_w-1].shape[0] * cropped[:,cell_w-1].shape[1]): # Recortar 5% derecha
        cropped = cropped[:,:int(cell_w-cell_w*0.05)]
        cell_w = cropped.shape[1]

    while count_black(cropped[0,:]) > 0.7*(cropped[0,:].shape[0] * cropped[0,:].shape[1]): # Recortar 5% arriba
        cropped = cropped[int(cell_h*0.05):,:]
        cell_h = cropped.shape[0]

    while count_black(cropped[cell_h-1,:]) > 0.7*(cropped[cell_h-1,:].shape[0] * cropped[cell_h-1,:].shape[1]): # Recortar 5% abajo
        cropped = cropped[:int(cell_h-cell_h*0.05),:]
        cell_h = cropped.shape[0]

    cropped_aux = np.copy(cropped)

    # Generar lineas aleatorias. Si no tocan pixeles negros, se puede recortar la imagen.

    for i in range(50):    
        for _ in range(iter):            
            if cell_w > 4:
                while True:
                    x = random.randint(0, cell_w - 2)
                    if x < cell_w/3 or x > 2*cell_w/3:
                        break

                if count_black(cropped[:,x:x+2]) == 0: # Linea vertical sin pixeles negros
                    if x < cell_w/2: # Recortar izquierda
                        cropped = cropped[:,x+1:]
                    else: # Recortar derecha
                        cropped = cropped[:,:x]
            
            if cell_h > 4:
                while True:
                    y = random.randint(0, cell_h - 2)
                    if y < cell_h/3 or y > 2*cell_h/3:
                        break
    
                if count_black(cropped[y:y+2,:]) == 0: # Linea horizontal sin pixeles negros
                    if y < cell_h/2: # Recortar arriba
                        cropped = cropped[y+1:,:]
                    else: # Recortar abajo
                        cropped = cropped[:y,:]
            
            cell_h, cell_w, _ = cropped.shape
        
        if (i < 49 and cell_h*cell_w < 40):
            cropped = cropped_aux
        else:
            break

    return cropped

# Contar el numero de pixeles negros de 'img'
def count_black(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    return np.sum(img_gray < 30)

# Contar el numero de pixeles blancos de 'img'
def count_white(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    return np.sum(img_gray > 225)

# Aplicar homografia (transformacion de perspectiva) para ajustar la perspectiva
# devuelve la celda transformada
def adjust_perspective(img, p1, p2, p3, p4):    
    # Definir los puntos de entrada y salida
    pts1 = np.float32([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]])
    pts2 = np.float32([[0, 0], [0, 100], [100, 100], [100, 0]])

    # Crear una mascara para la region de interes
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(mask, [pts1.astype(int)], (255, 255, 255))

    # Calcular la matriz de transformacion de perspectiva
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # Aplicar la transformacion a 'img'
    result = cv2.warpPerspective(img, M, (100, 100))

    return result

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

    # Escalar la imagen a las dimensiones originales (para verlas grandes en el explorador de archivos)
    #pixelated_img = cv2.resize(pixelated_img, (500, 500), interpolation=cv2.INTER_NEAREST)

    return pixelated_img

# Calcula la probabilidad de que 'img' pertenezca a cada clase
# devuelve una tupla [('prob[X]', 'prob[O]')]
def class_probability_xo(img, votes, dark_threshold):
    probability = [None, None]

    for iter in range(2):        
        assert img.shape[0] * img.shape[1] == votes[iter].size

        pixels = img.shape[0]

        prob_accumulated = 0

        for i in range(pixels):
            for j in range(pixels):
                if img[i, j] < dark_threshold: # Pixel negro
                    prob_accumulated += votes[iter][i][j] # Se acumula el valor del voto de ese pixel
                
        probability[iter] = prob_accumulated

    return probability

# Muestra por consola el tablero representado en 'board'
def print_board(board):
    print("     |     |     ")
    print("  " + board[0][0].lower() + "  |  " + board[0][1].lower() + "  |  " + board[0][2].lower() + "  ")
    print("_____|_____|_____")
    print("     |     |     ")
    print("  " + board[1][0].lower() + "  |  " + board[1][1].lower() + "  |  " + board[1][2].lower() + "  ")
    print("_____|_____|_____")
    print("     |     |     ")
    print("  " + board[2][0].lower() + "  |  " + board[2][1].lower() + "  |  " + board[2][2].lower() + "  ")
    print("     |     |     ")
    print("\n\n")

# Decide un movimiento para el bot en dificultad Facil
# devuelve el tablero con el nuevo movimiento y las coordenadas de su movimiento
def move_bot_easy(board, bot_symbol):    
    coords_empty = []

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                coords_empty.append([i,j])
    
    # Jugada aleatoria
    move = random.randint(0, len(coords_empty)-1)

    move_board = board
    move_board[coords_empty[move][0]][coords_empty[move][1]] = bot_symbol

    return move_board, coords_empty[move]

# Decide un movimiento para el bot en dificultad Intermedia
# devuelve el tablero con el nuevo movimiento y las coordenadas de su movimiento
def move_bot_intermediate(board, bot_symbol):
    # Puede ganar en el siguiente movimiento
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = bot_symbol
                if check_win(board) == bot_symbol:
                    return board, [i,j]
                else:
                    board[i][j] = ' '  # Deshacer movimiento

    # Puede bloquear la victoria del jugador
    player_symbol = 'x' if bot_symbol == 'o' else 'o'
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = player_symbol
                if check_win(board) == player_symbol:
                    board[i][j] = bot_symbol
                    return board, [i,j]
                else:
                    board[i][j] = ' '  # Deshacer movimiento

    # Jugada aleatoria
    return move_bot_easy(board, bot_symbol)

# Decide un movimiento para el bot en dificultad Pesadilla
# devuelve el tablero con el nuevo movimiento y las coordenadas de su movimiento
def move_bot_nightmare(board, bot_symbol):
    best_score = float('-inf')
    best_move = None

    player_symbol = 'x'
    if bot_symbol == 'x':
        player_symbol = 'o'

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = bot_symbol
                score = minimax(board, 0, False, bot_symbol, player_symbol)
                board[i][j] = ' '  # Deshacer el movimiento

                if score > best_score:
                    best_score = score
                    best_move = [i, j]

    # Realizar el mejor movimiento
    board[best_move[0]][best_move[1]] = bot_symbol
    return board, [best_move[0],best_move[1]]

# Algoritmo minimax para el bot de dificultad Pesadilla
def minimax(board, depth, is_maximizing, bot_symbol, player_symbol):
    scores = {player_symbol: -1, bot_symbol: 1, 'tie': 0}

    result = check_win(board)
    if result != '-':
        return scores[result]

    if is_maximizing:
        best_score = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = bot_symbol
                    score = minimax(board, depth + 1, False, bot_symbol, player_symbol)
                    board[i][j] = ' '  # Deshacer el movimiento
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = player_symbol
                    score = minimax(board, depth + 1, True, bot_symbol, player_symbol)
                    board[i][j] = ' '  # Deshacer el movimiento
                    best_score = min(score, best_score)
        return best_score

# Comprueba si hay un ganador en 'board'
# devuelve el simbolo del ganador, 'tie' en caso de empate, o '-' en otro caso
def check_win(board):
    for i in range(3):
        if (board[i][0] == 'x' and board[i][1] == 'x' and board[i][2] == 'x'):
            return 'x'
        elif (board[i][0] == 'o' and board[i][1] == 'o' and board[i][2] == 'o'):
            return 'o'
        elif (board[0][i] == 'x' and board[1][i] == 'x' and board[2][i] == 'x'):
            return 'x'
        elif (board[0][i] == 'o' and board[1][i] == 'o' and board[2][i] == 'o'):
            return 'o'
        
    if (board[0][0] == 'x' and board[1][1] == 'x' and board[2][2] == 'x'):
        return 'x'
    elif (board[0][0] == 'o' and board[1][1] == 'o' and board[2][2] == 'o'):
        return 'o'
    if (board[0][2] == 'x' and board[1][1] == 'x' and board[2][0] == 'x'):
        return 'x'
    elif (board[0][2] == 'o' and board[1][1] == 'o' and board[2][0] == 'o'):
        return 'o'          

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                return '-'
        
    return "tie"
    
# devuelve las coordenadas de los 3 puntos del simbolo ganador en 'board'
def winner_points(board):
    # Comprobar filas
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != ' ':
            return [[i, 0], [i, 1], [i, 2]]

    # Comprobar columnas
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] and board[0][j] != ' ':
            return [[0, j], [1, j], [2, j]]

    # Comprobar diagonales
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return [[0, 0], [1, 1], [2, 2]]
    elif board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return [[0, 2], [1, 1], [2, 0]]

    # No hay ganador
    return None

# devuelve las coordenadas (i,j) de los 4 puntos de una celda, siendo 'pos' el punto superior izquierdo
def coordinates(pos):
    i = pos[0]
    j = pos[1]
    p1 = (i, j)
    p2 = (i+1, j)
    p3 = (i+1, j+1)
    p4 = (i, j+1)
    return p1, p2, p3, p4


# FUNCIONES DE LA INTERFAZ GRAFICA

def play_easy_mode():
    global gamemode, root
    gamemode = "easy"
    root.destroy()

def play_intermediate_mode():
    global gamemode, root
    gamemode = "intermediate"
    root.destroy()

def play_nightmare_mode():
    global gamemode, root
    gamemode = "nightmare"
    root.destroy()

def register_mode():
    global gamemode, root
    gamemode = "register"
    root.destroy()

def quit():
    global cap
    # Liberar la conexion y cerrar la ventana
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    root.destroy()
    exit()

def toggle_save_images_disk():
    global root, save_images_disk, save_images_button

    if save_images_disk:
        save_images_button.configure(text = "Guardar archivos:\nOFF")
        save_images_disk = False
    else:
        save_images_button.configure(text = "Guardar archivos:\nON")
        save_images_disk = True

# La ventana sirve para leer y actualizar el fichero con la informacion de configuracion de ip/puerto
def window_ip_port():
    global config_file_path

    ip_port_window = tk.Toplevel(root)
    ip_port_window.title("IP/Port configuration")
    
    label = tk.Label(ip_port_window, text="Introduzca la IP y su puerto de DroidCam:", font=("Helvetica", 12))
    label.grid(row=0, column=0, columnspan=2, padx=20, pady=20)
    
    label_ip = tk.Label(ip_port_window, text="IP:")
    label_ip.grid(row=1, column=0, padx=10, pady=5, sticky=tk.E)

    entry_ip = tk.Entry(ip_port_window)
    entry_ip.grid(row=1, column=1, padx=(0,10), pady=5)
    
    label_port = tk.Label(ip_port_window, text="Port:")
    label_port.grid(row=2, column=0, pady=(5,20), sticky=tk.E)

    entry_port = tk.Entry(ip_port_window)
    entry_port.grid(row=2, column=1, padx=(0,10), pady=(5,10))
    
    with open(config_file_path, "r") as file:                        
        lines = file.readlines()

        # Obtener la IP y el puerto
        file_ip = lines[0].strip()
        file_port = lines[1].strip()

        # Llenar los entry con la IP y el puerto
        entry_ip.delete(0, tk.END)
        entry_ip.insert(0, file_ip)

        entry_port.delete(0, tk.END)
        entry_port.insert(0, file_port)
        
    def guardar_cambios():
        global ip, port

        new_ip = entry_ip.get()
        new_port = entry_port.get()

        # Leer el archivo
        with open(config_file_path, "r") as file:
            lines = file.readlines()
        
        lines[0] = f"{new_ip}\n"
        lines[1] = f"{new_port}"

        # Actualizar el archivo
        with open(config_file_path, "w") as file:
            file.writelines(lines)
        
        ip = new_ip
        port = new_port

        # Cerrar la ventana tras guardar los cambios
        ip_port_window.destroy()
    
    # Boton para guardar los cambios
    save_button = tk.Button(ip_port_window, text="Guardar Cambios", command=guardar_cambios)
    save_button.grid(row=3, column=0, columnspan=2, pady=(0, 20))
            
def help():
    # Crear la ventana
    help_window = tk.Tk()
    help_window.title("Ultimate Computer Vision TicTacToe")

    # Crear un widget de texto
    text = """# Proyecto 'Ultimate Computer Vision TicTacToe'
# Objetivo: simular un jugador de tres en raya, cuyos ojos son la camara de un dispositivo movil.
# Inicio: 29/11/2023

# Instalacion:
    # Seguir la guia de instalacion detallada en 'install.txt'.    

# Modos de juego: Facil, Medio, Pesadilla.

# Procedimiento:
    # Conectar el dispositivo movil y el ordenador a la misma red Wi-Fi (o conectarse desde el ordenador a los datos moviles del telefono).
    # Abrir DroidCam en el dispositivo movil.
    # Ejecutar este archivo, o el archivo 'U-CV-TTT.pyw'
    # Hacer clic en el boton 'IP/Port'
    # Introducir la IP y el puerto proporcionados por DroidCam en la ventana de 'IP/Port'.
    # Seleccionar un modo de juego.
    # Capturar imagenes con la barra espaciadora. Salir con 'q'.

# Tips:
    # Intentar usar folios blancos y rotulador negro.
    # Cuadrar el tablero de tres en raya aproximadamente en la cuadricula proporcionada, pero no acercarlo demasiado.
    # Puede dibujar o no dibujar los movimientos del bot en el papel. Es indiferente.

# Avanzado:
    # Puede entrenar su propio clasificador de simbolos con el archivo 'train_classifier.py'.
    # Puede probar el clasificador de simbolos con el archivo 'test_classifier.py'.

# Made with <3 by Macoris Decena Gimenez
# https://github.com/macorisd

# Para reportar bugs y errores: macorisd@gmail.com"""

    text_widget = tk.Text(help_window, wrap="word", width=110, height=33)
    text_widget.insert(tk.END, text)
    text_widget.configure(state="disabled", font=("Arial", 12))
    text_widget.pack()
    
    help_window.mainloop()













######################### MAIN

def main():    
    global cap, script_directory, config_file_path, save_images_disk, ip, port, root, save_images_button, gamemode, image_x_highlighted, image_x, image_o_highlighted, image_o

    #-- VARIABLES PARA PRUEBAS ------# 
    video_procesamiento = False      #
    
    cells_interface = False          #    
    #--------------------------------#
    if video_procesamiento:          #
        save_images_disk = False     #
    #--------------------------------#
    
    
    cap = None

    # Obtener el directorio del script actual
    script_directory = os.path.dirname(os.path.abspath(__file__)).replace('\\', '\\\\')

    # Booleano para guardar/no guardar capturas de las partidas en disco
    save_images_disk = False

    config_file_path = script_directory + "\\Config\\config.txt"

    # Cargar ip y puerto

    with open(config_file_path, "r") as file:                
        # Leer las lineas del archivo
        lines = file.readlines()

        # Obtener la IP y el puerto de las dos lineas
        ip = lines[0].strip()
        port = lines[1].strip()
    
    # Cargar imagenes

    image_x_highlighted = Image.open(script_directory + "\\Resources\\x_highlighted.png")
    image_x = Image.open(script_directory + "\\Resources\\x.png")
    image_o_highlighted = Image.open(script_directory + "\\Resources\\o_highlighted.png")
    image_o = Image.open(script_directory + "\\Resources\\o.png")
    
    # Margenes para el marco (Fase 0)

    dist_v = 50     # Margen vertical
    dist_h = 135    # Margen horizontal

    # Umbrales para escoger las lineas del tablero (Fase 1)

    orientation_threshold = 70 # Umbral para desechar lineas diagonales
    closeness_threshold = 50 # Umbral para desechar lineas repetidas (paralelas y muy juntas)

    # Parametros para el clasificador (Fase 4)

    pixels = 16
    dark_threshold = 250

    votes_xo = np.zeros((2, pixels, pixels), dtype=int)    

    classifier_file_path = ''

    # Lectura de ficheros con los resultados del clasificador (Fase 4)
    # en votes_xo[0] esta la informacion de X, y en votes_xo[1] esta la informacion de O

    for iter in range(2): # Iter. 1 : [X] | Iter. 2: [O]
        if (iter == 0):
            classifier_file_path = script_directory + "\\Classifier\\class_x.txt"            
        else:
            classifier_file_path = script_directory + "\\Classifier\\class_o.txt"

        rows_list = []
        
        with open(classifier_file_path, "r") as file:
            for line in file:                
                values = line.strip().split(",")
                                
                row = list(map(int, values))
                rows_list.append(row)
        
        votes_xo[iter] = np.array(rows_list) 

    # Informacion para equilibrar el clasificador (Fase 4)

    max_probs = [0, 0]    

    for i in range(pixels):
        for j in range(pixels):
            max_probs[0] += votes_xo[0][i][j]
            max_probs[1] += votes_xo[1][i][j]

    max_prob_class = 0

    if (max_probs[1] > max_probs[0]):
        max_prob_class = 1


    # Bucle principal. Cada iteracion es una visita al menu principal.

    while True:
        try:            
            cv2.destroyAllWindows() # Cerrar las ventanas de partidas anteriores

            if cap is not None:
                cap.release() # Liberar la captura de video

            # Interfaz grafica del menu principal

            root = tk.Tk()
            root.title("U-CV-TTT")
            root.configure(bg="#C2B7FF")

            root.iconbitmap(script_directory + "\\Resources\\icon.ico")

            image_logo = PhotoImage(file=script_directory + "\\Resources\\logo.png")

            img_label = tk.Label(root, image=image_logo)            
            img_label.grid(row=0, column=0, columnspan=3, padx=0, pady=0)

            label = tk.Label(root, text="Modos de juego", font=("Helvetica", 25), fg="#FFFFFF", bg="#C2B7FF")        
            label.grid(row=1, column=0, columnspan=3, padx=30, pady=30)

            gamemode = ""

            # Boton para jugar modo Facil
            play_button_easy = tk.Button(root, text="    Facil    ", font=("Helvetica", 13), fg="#55ed65", command=play_easy_mode)            
            play_button_easy.grid(row=2, column=0, padx=5, pady=0, sticky=tk.EW)

            # Boton para jugar modo Intermedio
            play_button_intermediate = tk.Button(root, text="Intermedio", font=("Helvetica", 13), fg="#ffb22e", command=play_intermediate_mode)
            play_button_intermediate.grid(row=2, column=1, padx=5, pady=0, sticky=tk.EW)

            # Boton para jugar modo Pesadilla
            play_button_nightmare = tk.Button(root, text="  Pesadilla  ", font=("Helvetica", 13), fg="#E90000", command=play_nightmare_mode)
            play_button_nightmare.grid(row=2, column=2, padx=5, pady=0, sticky=tk.EW)

            ip_port_button = tk.Button(root, text="IP/Port", font=("Helvetica", 13), fg="#000000", command=window_ip_port)
            ip_port_button.grid(row=3, column=1, padx=5, pady=(10,0), sticky=tk.EW)

            # Boton para activar/desactivar el guardado de archivos (capturas de las partidas)
            if (save_images_disk):
                save_images_button = tk.Button(root, text="Guardar archivos:\nON", font=("Helvetica", 10), fg="#000000", command=toggle_save_images_disk)
            else:
                save_images_button = tk.Button(root, text="Guardar archivos:\nOFF", font=("Helvetica", 10), fg="#000000", command=toggle_save_images_disk)

            save_images_button.grid(row=4, column=1, padx=0, pady=10, sticky=tk.EW)

            # Boton para ayuda
            help_button = tk.Button(root, text="Ayuda", font=("Helvetica", 10), fg="#000000", command=help)
            help_button.grid(row=5, column=1, padx=0, pady=5, sticky=tk.EW)

            # Boton para salir
            exit_button = tk.Button(root, text="Salir", font=("Helvetica", 10), fg="#9e0000", command=quit)
            exit_button.grid(row=6, column=1, padx=0, pady=(5,10))

            # Salir con la cruz de la ventana, o presionando 'q'
            root.protocol("WM_DELETE_WINDOW", quit)  
            root.bind("q", exit )

            root.mainloop()

            # Inicializar variables para la partida

            current_state = None

            board = [[' ' for _ in range(3)] for _ in range(3)] # Array 3x3 de caracteres que representa el contenido del tablero
            game = [[None, None] for _ in range(10)]
            
            turn = 0

            player_symbol = '-'
            bot_symbol = '-'

            # Configurar la conexion con el dispositivo para capturar video
            # herramienta usada: DroidCam (disponible en Play Store)

            cap = cv2.VideoCapture(f"http://{ip}:{port}/video")

            # Otras opciones:
            #device_id = 'f354ed7c1022' #adb
            #cap = cv2.VideoCapture(f'adb:{device_id}') #adb
            #cap = cv2.VideoCapture(1) #IVCam, IriunWebcam

            if not cap.isOpened():                
                messagebox.showerror("Error de conexion","Error al conectar con el dispositivo.")
                exit()

            # Bucle de partidas. Cada iteracion es un fotograma capturado.

            while True:

                # FASE 0: CAPTURA DEL FOTOGRAMA                        

                # Capturar el fotograma del video
                success, frame = cap.read()
                if success == False:                    
                    messagebox.showerror("Error de conexion","Error en la recepcion del video.")
                    exit()

                # Recortar marca de agua de DroidCam
                frame = frame[11:,:]    

                # Dibujar cuadrado para enfocar el tablero
                frame_sq = np.copy(frame)    
                cropped_frame = add_square(frame_sq, dist_v, dist_h) # Fotograma recortado para procesar

                # En la primera iteracion, define current_state como una matriz con la misma dimension de frame
                if current_state is None:
                    current_state = np.zeros_like(frame)
                    
                # Mostrar el video en tiempo real en la parte izquierda de la ventana
                # Mostrar fotograma capturado y procesado en la parte derecha de la ventana
                show_window(frame_sq,current_state)

                # Cheatsheet
                    # frame: fotograma original
                    # frame_sq: fotograma original + cuadrado
                    # cropped_frame: 'frame_sq' recortado (cuadrado) + binarizado
                    # cropped_frame_smooth: 'cropped_frame' suavizado y usado para obtener la matriz 'lines'
                    # cropped_frame_lines: 'cropped_frame' con las lineas elegidas + con las celdas detectadas
                    # cropped_frame_white: imagen plana blanca con el tamaño de 'cropped_frame'
                    # current_state: 'cropped_frame_lines' sobre 'frame'

                    # intersections: puntos de corte de las lineas del tablero (idealmente 8 lineas, 16 puntos)
                
                    # board: array de caracteres 3x3 que almacena el tablero
                    # game: array de 10 tuplas que contienen [board, cropped_frame_lines] (tableros e imagenes que componen una partida)

                key = cv2.waitKey(1)
                
                # 'Barra espaciadora' para procesar un fotograma de la partida
                if key & 0xFF == ord(' ') or video_procesamiento:
                    turn += 1  # Se ha ejecutado un turno

                    # FASE 1: PROCESAMIENTO DE LA CAPTURA Y DETECCION DE LINEAS

                    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY) # Conversion de BGR a escala de grises                  
                    _,cropped_frame = cv2.threshold(cropped_frame, 130, 255, cv2.THRESH_BINARY)  # Binarizacion para eliminar informacion desechable   
                    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_GRAY2BGR) # Conversion de escala de grises a BGR
                    add_square(cropped_frame, 0, 0) # Dibujar cuadricula para encajar el tablero
                    
                    cropped_frame_smooth = gaussian_smoothing(cropped_frame,2,5) # Suavizado gaussiano
                    edges = cv2.Canny(cropped_frame_smooth,14,220, apertureSize=3) # Deteccion de edges con Canny (tablero)
                    lines = cv2.HoughLines(edges,rho=0.5,theta=np.pi/180,threshold=41) # Deteccion de lineas con transformada de Hough

                    cropped_frame_lines = np.copy(cropped_frame)
                    cropped_frame_white = np.ones_like(cropped_frame) * 255

                    game[0][0] = [[' ', ' ', ' '] for _ in range(3)] # La partida empieza con un tablero vacio
                    game[0][1] = cropped_frame_white 


                    # FASE 2: FILTRADO DE LAS LINEAS PARA ESCOGER LAS 4 LINEAS DEL TABLERO                

                    if lines is not None:

                        # Almacenar primero las 4 lineas del marco exterior                

                        lines_vh_cartesian = [] # Array con las 8 lineas de interes (4 exteriores + 4 interiores)

                        height, width, _ = cropped_frame.shape
                        #print("width: " + str(width) + "; height: " + str(height)) # control

                        # El formato de las lineas es ((punto),(punto),orientacion)
                        left = ((0,-700),(0,700),'v')
                        right = ((width,-700),(width,700),'v')
                        up = ((-700,0),(700,0),'h')
                        down = ((-700,height),(700,height),'h')

                        lines_vh_cartesian.append(left)
                        lines_vh_cartesian.append(right)
                        lines_vh_cartesian.append(up)
                        lines_vh_cartesian.append(down)

                        inters_lrud = [None for _ in range(4)] # Intersecciones de una linea con la linea exterior izquierda, derecha, arriba y abajo

                        n_ver = 2 # Numero de lineas verticales
                        n_hor = 2 # Numero de lineas horizontales

                        # Analizar cada linea detectada para quedarme con las que me interesan

                        for line in lines: 
                            rho = line[0][0]
                            theta = line[0][1]
                            a = math.cos(theta)
                            b = math.sin(theta)
                            x0 = a * rho
                            y0 = b * rho
                            x1 = int(x0 + 2*width * (-b))
                            y1 = int(y0 + 2*height * (a))
                            x2 = int(x0 - 2*width * (-b))
                            y2 = int(y0 - 2*height * (a))
                            
                            descartar = False
                            orientation = ''

                            current_line = ((x1,y1), (x2,y2), orientation)

                            inters_lrud[0] = find_intersections((current_line, left)) # Interseccion con la linea exterior izquierda
                            if inters_lrud[0] is not None:
                                inters_lrud[1] = find_intersections((current_line, right)) # Interseccion con la linea exterior derecha
                                if inters_lrud[1] is None:                            
                                    continue
                                else: # La linea actual es horizontal
                                    orientation = 'h'
                                    #print("Intersecciones con izquerda: " + str(len(inters_lrud[0]))) # control
                                    x1 = int(inters_lrud[0][0][0]) 
                                    y1 = int(inters_lrud[0][0][1])
                                    #print("Intersecciones con derecha: " + str(len(inters_lrud[1]))) # control
                                    x2 = int(inters_lrud[1][0][0])
                                    y2 = int(inters_lrud[1][0][1])
                            else:
                                inters_lrud[2] = find_intersections((current_line, up)) # Interseccion con la linea exterior arriba
                                if inters_lrud[2] is not None:
                                    inters_lrud[3] = find_intersections((current_line, down)) # Interseccion con la linea exterior abajo
                                    if inters_lrud[3] is None:                                 
                                        continue
                                    else: # La linea actual es vertical
                                        orientation = 'v'
                                        #print("Intersecciones con arriba: " + str(len(inters_lrud[2]))) # control
                                        x1 = int(inters_lrud[2][0][0])
                                        y1 = int(inters_lrud[2][0][1])
                                        #print("Intersecciones con abajo: " + str(len(inters_lrud[3]))) # control
                                        x2 = int(inters_lrud[3][0][0])
                                        y2 = int(inters_lrud[3][0][1])                                                            

                            # Filtrado 1: Desechar las lineas con demasiada inclinacion (diagonales)
                            # Filtrado 2: Desechar las lineas repetidas (muy juntas)

                            if (orientation == 'v' and abs(x1-x2) < orientation_threshold): # Filtrado 1
                                for line_vh_c in lines_vh_cartesian: # Lineas ya definitivamente escogidas
                                    if line_vh_c[2] == 'v' and (abs(x1-line_vh_c[0][0]) < closeness_threshold or abs(x2-line_vh_c[1][0]) < closeness_threshold): # Filtrado 2
                                        descartar = True
                                        #cv2.line(cropped_frame_lines, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_AA) # control
                                        #cv2.line(cropped_frame_lines, (int(x2-(orientation_threshold)), y2), (int(x2+(orientation_threshold)), y2), (0, 255, 0), 3, cv2.LINE_AA) # control
                                        #cv2.line(cropped_frame_lines, (x2, int(y2-(orientation_threshold))), (x2, int(y2+(orientation_threshold))), (0, 255, 0), 3, cv2.LINE_AA) # control
                                        #cv2.line(cropped_frame_lines, (int(x1-(orientation_threshold)), y1), (int(x1+(orientation_threshold)), y1), (0, 255, 0), 3, cv2.LINE_AA) # control
                                        #cv2.line(cropped_frame_lines, (x1, int(y1-(orientation_threshold))), (x1, int(y1+(orientation_threshold))), (0, 255, 0), 3, cv2.LINE_AA) # control
                                        break
                            
                            elif (orientation == 'h' and abs(y1-y2) < orientation_threshold): # Filtrado 1
                                for line_vh_c in lines_vh_cartesian: # Lineas ya definitivamente escogidas
                                    if line_vh_c[2] == 'h' and (abs(y1-line_vh_c[0][1]) < closeness_threshold or abs(y2-line_vh_c[1][1]) < closeness_threshold): # Filtrado 2
                                        descartar = True
                                        #cv2.line(cropped_frame_lines, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_AA) # control
                                        #cv2.line(cropped_frame_lines, (int(x2-(orientation_threshold)), y2), (int(x2+(orientation_threshold)), y2), (0, 255, 0), 3, cv2.LINE_AA) # control
                                        #cv2.line(cropped_frame_lines, (x2, int(y2-(orientation_threshold))), (x2, int(y2+(orientation_threshold))), (0, 255, 0), 3, cv2.LINE_AA) # control
                                        #cv2.line(cropped_frame_lines, (int(x1-(orientation_threshold)), y1), (int(x1+(orientation_threshold)), y1), (0, 255, 0), 3, cv2.LINE_AA) # control
                                        #cv2.line(cropped_frame_lines, (x1, int(y1-(orientation_threshold))), (x1, int(y1+(orientation_threshold))), (0, 255, 0), 3, cv2.LINE_AA) # control
                                        break

                            else:
                                #cv2.line(cropped_frame_lines, (x1, y1), (x2, y2), (255, 0, 255), 3, cv2.LINE_AA) # control
                                #cv2.line(cropped_frame_lines, (int(x2-(orientation_threshold)), y2), (int(x2+(orientation_threshold)), y2), (0, 255, 0), 3, cv2.LINE_AA) # control
                                #cv2.line(cropped_frame_lines, (x2, int(y2-(orientation_threshold))), (x2, int(y2+(orientation_threshold))), (0, 255, 0), 3, cv2.LINE_AA) # control
                                #cv2.line(cropped_frame_lines, (int(x1-(orientation_threshold)), y1), (int(x1+(orientation_threshold)), y1), (0, 255, 0), 3, cv2.LINE_AA) # control
                                #cv2.line(cropped_frame_lines, (x1, int(y1-(orientation_threshold))), (x1, int(y1+(orientation_threshold))), (0, 255, 0), 3, cv2.LINE_AA) # control
                                continue

                            if not descartar: # La linea actual ha sido definitivamente escogida
                                cv2.line(cropped_frame_lines, (x1, y1), (x2, y2), (255, 183, 194), 3, cv2.LINE_AA)
                                lines_vh_cartesian.append(((x1, y1), (x2, y2), orientation)) # Idealmente habra 8 lineas escogidas
                                if orientation == 'v':
                                    n_ver += 1
                                elif orientation == 'h':
                                    n_hor += 1
                        

                        # FASE 3: SEGMENTACION DE LAS CELDAS DEL TABLERO

                        #print("Lineas: " + str(len(lines_vh_cartesian))) # control
                        #print(lines_vh_cartesian) # control

                        height, width, _ = frame.shape
                        current_state = frame

                        # (En este punto deberia haber 8 lineas detectadas)
                        if (len(lines_vh_cartesian)) == 8:
                            intersections = find_intersections(lines_vh_cartesian)
                            
                            #print("Intersecciones: " + str(len(intersections))) # control
                            #print(intersections) # control

                            # (En este punto deberia haber 4 lineas verticales y 4 lineas horizontales)
                            if (n_hor == 4 and n_ver == 4):

                                # (En este punto deberia haber 16 intersecciones detectadas)                
                                if (len(intersections) == 16):

                                    # Ordenar las intersecciones (puntos) para poder almacenarlos en un array 4x4 espacialmente ordenado

                                    intersections = sorted(intersections, key=lambda point: (point[1])) # Ordenar puntos segun coordenada y

                                    intersections = [intersections[i:i+4] for i in range(0, len(intersections), 4)] # Convertir el array 1x16 en un array 4x4

                                    for i in range (4): # Ordenar puntos de cada fila segun coordenada x
                                        intersections[i] = sorted(intersections[i], key=lambda point: (point[0]))                

                                    cells = [[None for _ in range(3)] for _ in range(3)] # Array 3x3 de imagenes (celdas)                                

                                    # Segmentar en 9 celdas y analizar

                                    for i in range(3):
                                        for j in range(3):
                                            # Dibujar celda en la imagen para mostrarla
                                            cropped_frame_lines = draw_cell(cropped_frame_lines, intersections, (i, j), (i+1, j), (i+1, j+1), (i, j+1))

                                            # Aplicar homografia para ajustar la perspectiva
                                            cells[i][j] = adjust_perspective(cropped_frame, intersections[i][j], intersections[i+1][j], intersections[i+1][j+1], intersections[i][j+1])

                                            #tam = cells[i][j].size # control
                                            #print("Tam: " + str(tam)) # control
                                        
                                            # Recortar la imagen de la celda segmentada para desechar partes blancas
                                            # 100 iteraciones del algoritmo. Es recomendable no bajar de 20 iteraciones.
                                            cells[i][j] = crop_cell(cells[i][j], 100)

                                            #tam = cells[i][j].size # control
                                            #print("Tam2: " + str(tam)) # control

                                            cell_h, cell_w, _ = cells[i][j].shape

                                            # Contar pixeles negros y blancos para calcular su proporcion

                                            black_pixels = count_black(cells[i][j])
                                            white_pixels = count_white(cells[i][j])
                                            if white_pixels == 0:
                                                white_pixels = 1
                                            ratio = black_pixels / white_pixels
                                                                                
                                            #print("Pixeles negros/pixeles blancos tras recorte: " + str(black_pixels) + "/" + str(white_pixels) + " = " + str(ratio)) # control  
                                            #print("Numero de pixeles: " + str(cells[i][j].size)) # control
                                            #print("Anchura: " + str(cell_w) + " | Altura: "+ str(cell_h) + " ||| Tamaño: " + str(cell_h*cell_w) + " ||| Ratio tamaño: " + str(max(cell_w,cell_h)/min(cell_w,cell_h))) # control

                                            # Decidir si la celda tiene una figura o esta vacia:
                                                # ancho * alto grande (si fuera pequeño seria porque cropped_cells habria recortado hasta el minimo)
                                                # ratio black_pixels/white_pixels razonable
                                                # la imagen resultante de crop_cells no es demasiado alargada (podria estar detectando un borde como figura)

                                            if (cell_h * cell_w >= 40) and (ratio > 0.03) and (ratio < 0.8) and not (cell_h > 3.75*cell_w or cell_w > 3.75*cell_h): # Celda con figura                                                

                                                # FASE 4: MOVIMIENTO DEL JUGADOR - CLASIFICACION DE LAS FIGURAS ENCONTRADAS 

                                                # Preparar la imagen para clasificarla                                                

                                                img_classify = np.copy(cells[i][j])
                                                img_classify = cv2.cvtColor(img_classify, cv2.COLOR_BGR2GRAY) # Convertir a escala de grises                                                                                            
                                                img_classify = to_square(img_classify) # Convertir imagen a cuadrado                                                
                                                img_classify = pixelate(img_classify, pixels) # Pixelar imagen

                                                # Calcular las probabilidades. prob[0] -> [X] | prob[1] -> [O]

                                                prob = class_probability_xo(img_classify, votes_xo, dark_threshold) 

                                                # Equilibrar las probabilidades:
                                                    # Regla de 3:
                                                    # maximo del menor -------- mi valor
                                                    # maximo del mayor -------- x

                                                prob[1-max_prob_class] = int(prob[1-max_prob_class] * max_probs[max_prob_class] / max_probs[1-max_prob_class])

                                                if board[i][j] not in ['x', 'o']: # No habia nada en esa posicion de board
                                                    if prob[0] > prob[1]:                                                                                                                
                                                        if player_symbol == 'o':
                                                            cropped_frame_lines = draw_xo_cell(cropped_frame_lines, 'x', False, intersections, (i, j), (i+1, j), (i+1, j+1), (i, j+1))
                                                            board[i][j] = 'x'
                                                        else:
                                                            board[i][j] = "X"
                                                            cropped_frame_lines = draw_xo_cell(cropped_frame_lines, 'x', False, intersections, (i, j), (i+1, j), (i+1, j+1), (i, j+1))
                                                    elif prob[1] > prob[0]:                                                        
                                                        if player_symbol == 'x':
                                                            cropped_frame_lines = draw_xo_cell(cropped_frame_lines, 'o', False, intersections, (i, j), (i+1, j), (i+1, j+1), (i, j+1))
                                                            board[i][j] = 'o'
                                                        else:
                                                            board[i][j] = "O"  
                                                            cropped_frame_lines = draw_xo_cell(cropped_frame_lines, 'o', False, intersections, (i, j), (i+1, j), (i+1, j+1), (i, j+1))
                                                
                                                    #print(f"Probabilidad de X: {prob[0]}\t\tProbabilidad de O: {prob[1]}\t\tClasificado como: {board[i][j]}") # control
                                                
                                                else:
                                                    # Dibujar simbolos
                                                    if board[i][j] == 'x':
                                                        cropped_frame_lines = draw_xo_cell(cropped_frame_lines, 'x', False, intersections, (i, j), (i+1, j), (i+1, j+1), (i, j+1))
                                                    elif board[i][j] == 'o':
                                                        cropped_frame_lines = draw_xo_cell(cropped_frame_lines, 'o', False, intersections, (i, j), (i+1, j), (i+1, j+1), (i, j+1))                                      

                                            else: # Celda vacia
                                                if board[i][j] not in ['x', 'o']:
                                                    board[i][j] = ' '

                                                # Dibujar simbolos
                                                elif board[i][j] == 'x':
                                                    cropped_frame_lines = draw_xo_cell(cropped_frame_lines, 'x', False, intersections, (i, j), (i+1, j), (i+1, j+1), (i, j+1))
                                                elif board[i][j] == 'o':
                                                    cropped_frame_lines = draw_xo_cell(cropped_frame_lines, 'o', False, intersections, (i, j), (i+1, j), (i+1, j+1), (i, j+1))

                                    game[turn][0] = [row.copy() for row in board] # Guardar tablero en el array de la partida
                                    game[turn][1] = np.copy(cropped_frame_lines) # Guardar imagen en el array de la partida


                                    if cells_interface:
                                        # Mostrar interfaz con las celdas recortadas
                                        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
                                        plain_gray = np.ones((50, 50, 3), dtype=np.uint8) * 100                    
                                        for i in range(3):
                                            for j in range(3):
                                                cell_h, cell_w, _ = cells[i][j].shape
                                                if cell_h * cell_w < 40:
                                                    axs[i, j].imshow(plain_gray, cmap='gray')
                                                else:
                                                    axs[i, j].imshow(cells[i][j], cmap='gray')
                                                axs[i, j].axis('off')
                                        
                                        fig.patch.set_facecolor('gray')
                                        plt.show()  


                                    # Mostrar movimiento
                                    current_state[dist_v:height-dist_v, dist_h:width-dist_h] = cropped_frame_lines # Superponer el cuadrado procesado encima de 'current_state'                                
                                    show_window(frame_sq,current_state) # Mostrar el movimiento del jugador
                                    cv2.waitKey(1)
                                    
                                    #print_board(board) # Dibujar tablero en consola

                                    # FASE 5: MOVIMIENTO DEL BOT - ANALISIS DEL TABLERO Y DECISION DEL MOVIMIENTO

                                    filled_cells = 0
                                    move_coords = [None, None]

                                    # Encontrar numero de casillas ocupadas y almacenar posicion del ultimo movimiento del jugador

                                    for i in range(3):
                                        for j in range(3):
                                            if board[i][j] in ['X', 'O', 'x', 'o']:
                                                filled_cells += 1

                                                if board[i][j] == 'X':
                                                    move_coords = [i,j]
                                                    board[i][j] = 'x'
                                                elif board[i][j] == 'O':
                                                    move_coords = [i,j]
                                                    board[i][j] = 'o'

                                    # Empieza a jugar el bot (1 captura hecha, 0 celdas ocupadas)
                                    if turn == 1 and filled_cells == 0:
                                        bot_symbol = 'o'
                                        player_symbol = 'o'
                                        turn -= 1
                                    
                                    # Empieza a jugar el jugador (1 captura hecha, 1 celda ocupada)
                                    elif turn == 1 and filled_cells == 1:
                                        player_symbol = board[move_coords[0]][move_coords[1]]
                                        bot_symbol = 'o'

                                        if player_symbol == 'o':
                                            bot_symbol = 'x'                                
                                    
                                    # turn indica cuantas celdas deberia haber ocupadas en ese turno
                                    # filled_cells indica cuantas celdas estan ocupadas en ese turno
                                    if (turn == filled_cells):
                                        # Se comprueba si hay ganador

                                        win = check_win(board)

                                        if (win == bot_symbol): # Gana el bot                                           
                                            for i in range(3):
                                                p1, p2, p3, p4 = coordinates(winner_points(board)[i])
                                                cropped_frame_lines = draw_xo_cell(cropped_frame_lines, bot_symbol, True, intersections, p1, p2, p3, p4) # Resaltar celdas ganadoras
                                            current_state[dist_v:height-dist_v, dist_h:width-dist_h] = cropped_frame_lines # Superponer el cuadrado procesado encima de 'current_state'                                
                                            show_window(frame_sq,current_state)
                                            cv2.waitKey(1)                     
                                            game[turn][0] = [row.copy() for row in board]
                                            game[turn][1] = np.copy(cropped_frame_lines)                   
                                            messagebox.showinfo("Fin de la partida", "¡Gana el bot!")
                                            break
                                        
                                        elif (win == player_symbol): # Gana el jugador
                                            for i in range(3):
                                                p1, p2, p3, p4 = coordinates(winner_points(board)[i])
                                                cropped_frame_lines = draw_xo_cell(cropped_frame_lines, player_symbol, True, intersections, p1, p2, p3, p4) # Resaltar celdas ganadoras
                                            current_state[dist_v:height-dist_v, dist_h:width-dist_h] = cropped_frame_lines # Superponer el cuadrado procesado encima de 'current_state'                                
                                            show_window(frame_sq,current_state)
                                            cv2.waitKey(1)
                                            game[turn][0] = [row.copy() for row in board] # Guardar tablero en el array de la partida
                                            game[turn][1] = np.copy(cropped_frame_lines) # Guardar imagen en el array de la partida
                                            messagebox.showinfo("Fin de la partida", "¡Has ganado!")
                                            break

                                        elif win == "tie": # Empate
                                            messagebox.showinfo("Fin de la partida", "¡Es un empate!")
                                            break                                                                        

                                        if turn < 9: # La partida aun no ha terminado. Mueve el bot.
                                            if gamemode == "easy":
                                                board, move_coords = move_bot_easy(board, bot_symbol)       
                                            elif gamemode == "intermediate":
                                                board, move_coords = move_bot_intermediate(board, bot_symbol)  
                                            elif gamemode == "nightmare":
                                                board, move_coords = move_bot_nightmare(board, bot_symbol)
                                            
                                            i = move_coords[0]
                                            j = move_coords[1]

                                            # Dibujar movimiento del bot

                                            if (bot_symbol == 'x'):                                            
                                                cropped_frame_lines = draw_xo_cell(cropped_frame_lines, 'x', False, intersections, (i, j), (i+1, j), (i+1, j+1), (i, j+1))
                                            
                                            elif bot_symbol == 'o':
                                                cropped_frame_lines = draw_xo_cell(cropped_frame_lines, 'o', False, intersections, (i, j), (i+1, j), (i+1, j+1), (i, j+1))

                                            turn += 1 # Se ha ejecutado un turno

                                            game[turn][0] = [row.copy() for row in board] # Guardar tablero en el array de la partida
                                            game[turn][1] = np.copy(cropped_frame_lines) # Guardar imagen en el array de la partida
                                            
                                            time.sleep(1) # El bot "piensa" su movimiento durante 1 segundo

                                            # Mostrar movimiento
                                            current_state[dist_v:height-dist_v, dist_h:width-dist_h] = cropped_frame_lines # Superponer el cuadrado procesado encima de 'current_state'                                
                                            show_window(frame_sq,current_state)
                                            cv2.waitKey(1)
                                            
                                            #print_board(board) # Dibujar tablero en consola

                                            # Se comprueba si hay ganador

                                            win = check_win(board)

                                            if (win == bot_symbol): # Gana el bot
                                                for i in range(3):
                                                    p1, p2, p3, p4 = coordinates(winner_points(board)[i])
                                                    cropped_frame_lines = draw_xo_cell(cropped_frame_lines, bot_symbol, True, intersections, p1, p2, p3, p4) # Resaltar celdas ganadoras
                                                current_state[dist_v:height-dist_v, dist_h:width-dist_h] = cropped_frame_lines # Superponer el cuadrado procesado encima de 'current_state'                                
                                                show_window(frame_sq,current_state)
                                                cv2.waitKey(1)
                                                game[turn][0] = [row.copy() for row in board]
                                                game[turn][1] = np.copy(cropped_frame_lines)
                                                messagebox.showinfo("Fin de la partida", "¡Gana el bot!")
                                                break
                                            
                                            elif (win == player_symbol): # Gana el jugador
                                                for i in range(3):
                                                    p1, p2, p3, p4 = coordinates(winner_points(board)[i])
                                                    cropped_frame_lines = draw_xo_cell(cropped_frame_lines, player_symbol, True, intersections, p1, p2, p3, p4) # Resaltar celdas ganadoras
                                                current_state[dist_v:height-dist_v, dist_h:width-dist_h] = cropped_frame_lines # Superponer el cuadrado procesado encima de 'current_state'                                
                                                show_window(frame_sq,current_state)
                                                cv2.waitKey(1)
                                                game[turn][0] = [row.copy() for row in board]
                                                game[turn][1] = np.copy(cropped_frame_lines)
                                                messagebox.showinfo("Fin de la partida", "¡Has ganado!")
                                                break

                                            elif win == "tie": # Empate
                                                messagebox.showinfo("Fin de la partida", "¡Es un empate!")
                                                break
                                        
                                        elif turn == 9:
                                            break
                                    
                                    elif turn > filled_cells or turn < filled_cells: # No se ha detectado un simbolo adicional en este turno, o se han detectado mas simbolos de la cuenta
                                        # Se retrocede al turno anterior, descartando la nueva deteccion erronea

                                        game[turn][0] = None
                                        game[turn][1] = None
                                        turn -= 1
                                        board = [row.copy() for row in game[turn][0]]

                                        current_state[dist_v:height-dist_v, dist_h:width-dist_h] = game[turn][1]
                                        show_window(frame_sq,current_state)
                                        cv2.waitKey(1)

                                        # Mostrar mensajes segun situacion

                                        if turn == 0:
                                            messagebox.showerror("Error en la partida", f"El programa ha detectado {filled_cells} simbolos cuando deberia haber 0 o 1. Por favor, vuelva a enfocar la imagen o dibuje simbolos mejores.")
                                        elif turn+1 > filled_cells:
                                            messagebox.showerror("Error en la partida", f"El programa ha detectado menos simbolos de los que deberia (deberia haber {turn+1}). Por favor, vuelva a enfocar la imagen o dibuje simbolos mejores.")
                                        elif turn+1 < filled_cells:
                                            messagebox.showerror("Error en la partida", f"El programa ha detectado mas simbolos de los que deberia (deberia haber {turn+1}). Por favor, vuelva a enfocar la imagen o dibuje simbolos mejores.")
                            
                                else: # Deberia haber 16 intersecciones
                                    turn -= 1
                                    board = game[turn][0]
                                    current_state[dist_v:height-dist_v, dist_h:width-dist_h] = game[turn][1]                               
                                    show_window(frame_sq,current_state)
                                    cv2.waitKey(1)
                                    messagebox.showerror("Error en la deteccion", f"El programa ha fallado en la deteccion de las intersecciones de las lineas del tablero. Deberia haber 16, y ha detectado {len(intersections)}. Por favor, vuelva a enfocar la imagen o dibuje un tabero mejor.")

                            else: # Deberia haber 4 lineas verticales y 4 horizontales
                                turn -= 1
                                board = game[turn][0]
                                current_state[dist_v:height-dist_v, dist_h:width-dist_h] = game[turn][1]                               
                                show_window(frame_sq,current_state)
                                cv2.waitKey(1)
                                messagebox.showerror("Error en la deteccion", f"El programa ha fallado en la deteccion de las lineas del tablero. Deberia haber 2 verticales y 2 horizontales. Por favor, vuelva a enfocar la imagen o dibuje un tablero mejor.")                            

                        else: # Deberia haber 8 lineas
                            turn -= 1
                            board = game[turn][0]
                            current_state[dist_v:height-dist_v, dist_h:width-dist_h] = game[turn][1]                               
                            show_window(frame_sq,current_state)
                            cv2.waitKey(1)
                            messagebox.showerror("Error en la deteccion", f"El programa ha fallado en la deteccion de las lineas del tablero. Deberia haber 4, y ha detectado {len(lines_vh_cartesian)-4}. Por favor, vuelva a enfocar la imagen o dibuje un tablero mejor.")
                    
                    else: # Deberia haber lineas
                        turn -= 1
                        board = game[turn][0]
                        current_state[dist_v:height-dist_v, dist_h:width-dist_h] = game[turn][1]                               
                        show_window(frame_sq,current_state)
                        cv2.waitKey(1)
                        messagebox.showerror("Error en la deteccion", f"El programa ha fallado en la deteccion de las lineas del tablero. Deberia haber 4, y ha detectado {len(lines_vh_cartesian)-4}. Por favor, vuelva a enfocar la imagen o dibuje un tablero mejor.")
                                        
                # 'q' para salir
                elif key & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break    

            # Guardar fotograma procesado
            if save_images_disk:
                frame_directory = script_directory + "\\" + datetime.now().strftime("Frames [%d-%m-%Y] [%H;%M;%S]")
                os.makedirs(frame_directory)

                for i in range(1,10):
                    frame_path = frame_directory + f"\\frame_{len(os.listdir(frame_directory))+1}.png"
                    if (game[i][1] is not None):
                        cv2.imwrite(frame_path, game[i][1])

        except Exception as e:            
            messagebox.showerror("Error", f"Se produjo una excepcion: {e}. Por favor, intentelo de nuevo.")                              

if __name__ == "__main__":
    main()