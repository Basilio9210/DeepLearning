

@author: Fernando Pamplona
"""
import numpy as np # para operar vectores y matrices
import wave        # Libreria para leer y esctribir archivos WAV
import struct      # Libreria que estructura información dividida
import matplotlib.pyplot as plt  # Libreria para graficar datos
import pyaudio     # Libreria para utilizar el audio del PC

# =============================================================================
# Sección de Lectura del archivo WAV
# =============================================================================
waveFile=wave.open('sound.wav','r')     # Lectura del archivo WAV dentro del objeto waveFile
waveInfo=waveFile.getparams()           # Obtiene los parámetros del archivo de audio
print (waveInfo)                        # Visualiza

length = waveFile.getnframes()          # Obtiene el tamaño en muestras del archivo WAV
frameRate= waveFile.getframerate()      # Obtiene muestras por segurno para la reproducción

muestras=int(length)                    # Muestras a utilizar en el Ejemplo

channel1=np.zeros((muestras))           # Declaración de vector para almacenar información del canal1
channel2=np.zeros((muestras))           # Declaración del vector para el canal #2

for i in range(0,muestras):             # Estructuración de muestra por muestra
    waveData = waveFile.readframes(1)   # y almacendao en las variables channel 1 y 2
    data = struct.unpack("<2h", waveData)
    channel1[i]=data[0]
    channel2[i]=data[1]
#    print (int(waveData[0]),"\t",int(waveData[1]),"\t",int(waveData[2]),"\t",int(waveData[3]),"\t",int(data[0]),"\t",int(data[1]))

# =============================================================================
# Seccion de cambios a la onda
# =============================================================================
c1=(channel1).astype(np.int16)      # Conversion de flotante a entero

MMM=20  #Muestras media movil
## Filtro FIR Medi Movil
#c1MediaMovil=np.zeros((muestras)).astype(np.float64)
#for i in range(MMM,muestras):
##    c1MediaMovil[i]=(c1[i-7]/8+c1[i-6]/8+c1[i-5]/8+c1[i-4]/8+c1[i-3]/8+c1[i-2]/8+c1[i-1]/8+c1[i]/8)
#    for j in range(i-MMM,i):
#        c1MediaMovil[i]+=c1[j]/MMM
#c1Filtrado=(c1MediaMovil).astype(np.int16)

#Filtro IIR Media Movil
c1IIR=np.zeros((muestras)).astype(np.float64)
for j in range(0,MMM-1):
    c1IIR[0]+=c1[j]/MMM
for i in range(MMM+1,muestras):
    c1IIR[i]=c1IIR[i-1]+(c1[i]/MMM-c1[i-MMM]/MMM)
c1Filtrado=(c1IIR).astype(np.int16)

# =============================================================================
# Sección de Graficado
# =============================================================================
plt.figure(1)   #crear una nueva figura
plt.plot(c1)    #plasmar los datos en la gráfica
plt.show()      #mostrar la gráfica

plt.figure(2)   #crear una nueva figura
plt.plot(c1Filtrado)    #plasmar los datos en la gráfica
plt.show()      #mostrar la gráfica

OriginalFFT=np.abs(np.fft.fft(c1))  #Extracción del espectro de potencia

plt.figure(3)                       #crear una nueva figura
plt.plot(OriginalFFT)               #plasmar los datos en la gráfica
plt.axis([0,40000,np.amin(OriginalFFT), np.amax(OriginalFFT)])   #establecer la sección de interés de la gráfica
plt.show()                          #mostrar

FiltradaFFT=np.abs(np.fft.fft(c1Filtrado))  #Extracción del espectro de potencia

plt.figure(4)                       #crear una nueva figura
plt.plot(FiltradaFFT)               #plasmar los datos en la gráfica
plt.axis([0,40000,np.amin(FiltradaFFT), np.amax(FiltradaFFT)])   #establecer la sección de interés de la gráfica
plt.show()                          #mostrar

# =============================================================================
# Seccion de reproducción de Audio
# =============================================================================
PyAudio = pyaudio.PyAudio # Definición del objeto para la salida de audio
p = PyAudio()                                       # Configuración del puerto de salida 
stream = p.open(format = p.get_format_from_width(2),# Resolucon 1:1Byte=8bits; 2:2bytes=16bits  
                    channels = 1,                   # Número de canales
                    rate = frameRate,               # Muestras por segundo
                    output = True)                  # Activar altavoz
stream.write(c1)        # Escribir los datos en el puerto de audio
stream.stop_stream()    # Terminar la escritura
stream.close()          # y
p.terminate()           # Destruir el objeto

PyAudio = pyaudio.PyAudio # Definición del objeto para la salida de audio
p = PyAudio()                                       # Configuración del puerto de salida 
stream = p.open(format = p.get_format_from_width(2),# Resolucon 1:1Byte=8bits; 2:2bytes=16bits  
                    channels = 1,                   # Número de canales
                    rate = frameRate,               # Muestras por segundo
                    output = True)                  # Activar altavoz
stream.write(c1Filtrado)        # Escribir los datos en el puerto de audio
stream.stop_stream()    # Terminar la escritura
stream.close()          # y
p.terminate()           # Destruir el objeto
