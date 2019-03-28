"""
Created on Sat Aug 25 13:14:28 2018

@author: Fernando Pamplona
"""
import numpy as np 	# Librería para operar vectores y matrices
import wave        	# Librería para leer y escribir archivos WAV
import struct      	# Librería que estructura información dividida
import pyaudio     	# Librería para utilizar el audio del PC


# =============================================================================
# Sección de lectura del archivo WAV
# =============================================================================
waveFile=wave.open('rebuzno.wav','r')   	# Lectura del archivo WAV dentro del objeto waveFile
waveInfo=waveFile.getparams()           		# Obtiene los parámetros del archivo de audio
print (waveInfo)                        			# Visualiza

length = waveFile.getnframes()       	# Obtiene el tamaño en muestras del archivo WAV
frameRate= waveFile.getframerate()   	# Obtiene muestras por segurno para la reproducción

muestras=int(length/2)                   	# Muestras a utilizar en el Ejemplo

channel1=np.zeros((muestras))           	# Declaración de vector para almacenar información del canal1

for i in range(0,muestras):             	# Estructuración de muestra por muestra
    waveData = waveFile.readframes(1)       # y almacenado en la variable channel1
    data = struct.unpack("<2h", waveData)
    channel1[i]=data[0]
    
c1=(channel1).astype(np.int16)          # Conversión de flotante a entero, uso de bits a tener en cuenta en PyAudio
# =============================================================================
# Seccion del tratamiento de la señal
# =============================================================================


# =============================================================================
# Seccion de reproducción de Audio
# =============================================================================
PyAudio = pyaudio.PyAudio 	# Definición del objeto para la salida de audio
p = PyAudio()                           # Configuración del puerto de salida 
stream = p.open(format = p.get_format_from_width(2),# Resolución (1):1Byte=8bits; (2):2bytes=16bits… 
                    channels = 1,       # Número de canales
                    rate = frameRate, 	# Muestras por segundo
                    output = True)      # Activar altavoz
stream.write(c1)        		# Escribir los datos en el puerto de audio
stream.stop_stream()            # Terminar la escritura
stream.close()          		# y
p.terminate()           		# Destruir el objeto
