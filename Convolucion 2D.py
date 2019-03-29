# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 20:22:19 2018

@author: ASUS
"""

import cv2 			# Para leer y mostrar la imágenes
import numpy  as np # para operar vectores y matrices

# =============================================================================
# Sección de Lectura del archivo jpg
# =============================================================================

img = cv2.imread('lena.jpg',0)  #leer la imágen

# =============================================================================
# Seccion de convolución
# =============================================================================

# Definir el Kernel o señal para la convolución (nombrarlo h1)


# Definir la variable donde se almacenará el resultado de la convolución (nombrarla y1)


#
imgPaded=np.pad(img,((1,1),(1,1)),'constant',constant_values=((0,0),(0,0)))

# Iniciar el algoritmo de la convolución. 
# (-invertir kernel - alinear las  señales- sumar los productos - desplazar las señales y repetir)

for i in range(1,np.size(img,0)):
    for j in range(1,np.size(img,1)):
		imgPaded[i,j]

# =============================================================================
# Seccion de muestra de imágenes
# =============================================================================		
y1min=np.min(y1)
y1max=np.max(y1)
y1=(y1-y1min)/y1max
filtrada=(y1*255).astype(np.uint8) 
cv2.imshow('Original',img)
cv2.imshow('Filtrada',filtrada)
cv2.waitKey(0)
cv2.destroyAllWindows()
