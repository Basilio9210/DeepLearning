"""
Created on Sat Aug 25 13:14:28 2018

@author: Fernando Pamplona
"""
import numpy as np # para operar vectores y matrices
import wave        # Libreria para leer y esctribir archivos WAV
import struct      # Libreria que estructura información dividida

# =============================================================================
# Sección de Lectura del archivo WAV
# =============================================================================
waveFile=wave.open('sound.wav','r')     # Lectura del archivo WAV dentro del objeto waveFile
waveInfo=waveFile.getparams()           # Obtiene los parámetros del archivo de audio
print (waveInfo)                        # Visualiza

length = waveFile.getnframes()          # Obtiene el tamaño en muestras del archivo WAV
frameRate= waveFile.getframerate()      # Obtiene muestras por segurno para la reproducción

muestras=500#int(length)                    # Muestras a utilizar en el Ejemplo

c1=np.zeros((muestras))           # Declaración de vector para almacenar información del canal1

for i in range(0,muestras):             # Estructuración de muestra por muestra
    waveData = waveFile.readframes(1)   # y almacendao en las variables channel 1 y 2
    data = struct.unpack("<2h", waveData)
    c1[i]=data[0]



# Definir el Kernel o señal para la convolución (nombrarlo h1)
h1=np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])

# Definir la variable donde se almacenará el resultado de la convolución


#
c1Paded=np.pad(c1,(h1.size-1,),'constant',constant_values=(0,))  # adicionar un pad de 0s

# Iniciar el algoritmo de la convolución. 
# (-invertir kernel - alinear las  señales- sumar los productos - desplazar las señales y repetir)

