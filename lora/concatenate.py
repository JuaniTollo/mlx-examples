import numpy as np
import os
def devolverUltimaPosicionNoNula(array):
    # Inicializa una lista para almacenar los valores deseados
    valores_finales_no_cero = []

    # Itera sobre cada uno de los vectores en el array
    for vector in array:
        # Filtra el vector para quedarte solo con los valores no cero
        valores_no_cero = vector[vector != 0]
        # Obtiene el último valor no cero, si existe
        if valores_no_cero.size > 0:
            ultimo_valor_no_cero = valores_no_cero[-1]
        else:
            ultimo_valor_no_cero = None  # O lo que consideres apropiado para indicar que no hay valores no cero
        # Agrega el valor a la lista
        valores_finales_no_cero.append(ultimo_valor_no_cero)
    
    # Muestra los resultados
    return np.array(valores_finales_no_cero)
def convertBatchMatrixToColumnMatrix(T):
    datos_aplanados = [elem for sublist in T for elem in sublist]

    # Crear una matriz de len(datos_aplanados) x 1
    matriz = []
    for i in range(len(datos_aplanados)):
        matriz.append([datos_aplanados[i]])
    return matriz

def concatenateTargetVectors():
    i = 0
    all_arrays = []
    while True:
        try:
            file_pattern = './experiments/results/targets/targets*.npy'
            file_pattern_temp = file_pattern.replace("*",f"{i}")
            loaded_arrays = np.load(file_pattern_temp)
            loaded_arrays_target = devolverUltimaPosicionNoNula(loaded_arrays)
            all_arrays.append(loaded_arrays_target)
            i+=1
        except:
            print(f"Termino luego de intentar concatenar el batch numero {i}")
            break
    T = np.stack(np.stack(all_arrays))

    matriz = convertBatchMatrixToColumnMatrix(T)

    return matriz
T = concatenateTargetVectors()
np.save(f'./experiments/results/targets.npy', T)

type(T)
import numpy as np

def devolverUltimaPosicionNoNulaLogits(array):
    # Inicializa una lista para almacenar los arrays resultantes
    resultados_finales = []
    
    # Itera sobre la primera dimensión del array
    for subarray in array:
        # Inicializa una lista para almacenar los últimos valores no cero de cada vector
        ultimos_valores_no_cero = []
        
        # Itera sobre cada vector en la segunda dimensión (n)
        for vector in subarray:
            # Identifica los índices de los elementos no cero
            indices_no_cero = np.nonzero(vector)[0]
            
            # Si hay valores no cero, obtiene el último de ellos; de lo contrario, usa un valor por defecto
            if indices_no_cero.size > 0:
                ultimo_no_cero = vector[indices_no_cero[-1]]
            else:
                ultimo_no_cero = np.nan  # Puedes usar np.nan o cualquier otro valor que indique "vacío" o "nulo"
            
            # Agrega el último valor no cero a la lista
            ultimos_valores_no_cero.append(ultimo_no_cero)
        
        # Agrega el resultado de este subarray a los resultados finales
        resultados_finales.append(ultimos_valores_no_cero)
    
    # Convierte los resultados finales en un array de NumPy antes de devolverlos
    return np.array(resultados_finales)

# Ejemplo de uso
# Supongamos que 'array' es tu array de dimensiones (4, n, 32000)
# array = np.random.rand(4, 10, 32000)  # Un ejemplo con datos aleatorios
# resultado = devolverUltimaPosicionNoNulaLogits(array)
# print(resultado.shape)  # Esto debería darte (4, n) como dimensión

def contar_archivos_en_directorio(ruta):
    # Obtener la lista de archivos en el directorio
    archivos = os.listdir(ruta)
    # Contar la cantidad de archivos
    cantidad_archivos = len(archivos)
    return cantidad_archivos

# Ruta del directorio que quieres verificar
directorio = './experiments/results/logits'
contar_archivos_en_directorio(directorio)
#una matriz tiene nxpxq dimensiones
def returnLastsNotNullLogits(M):
    L = []
    for i in range(0,4):
        p = M.shape[1]-1
        while np.all(M[i,p,:]==0):
            p -= 1
        L_i = M[i,p,:]
        L.append(L_i)
    L = np.stack(L)
    return L

def concatenateLogitsMatrix():
    i = 0
    L = []
    directorio = './experiments/results/logits'
    b = contar_archivos_en_directorio(directorio)

    for i in range(0,b):
        file_pattern = './experiments/results/logits/logits*.npy'
        file_pattern_temp = file_pattern.replace("*",f"{i}")
        loaded_arrays = np.load(file_pattern_temp)
        
        loaded_arrays_target = returnLastsNotNullLogits(loaded_arrays)
        
        L.append(loaded_arrays_target)
        i+=1
    
    L = np.array(L)
    # Concatenar todas las matrices de la lista a lo largo del eje de las filas (eje 0)
    matriz_combinada = np.concatenate(L, axis=0)
    return matriz_combinada
    
M = concatenateLogitsMatrix()
np.save(f'./experiments/results/logits.npy', M)
