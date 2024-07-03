import cv2
import os
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm
import random as rn
import pickle 
import matplotlib.pyplot as plt
DATADIR=r"Naranjas"
CATEGORIAS=[""]
tamaño_img=1080

def generar_datos():
    data=[]
    for categoria in CATEGORIAS:
        carpeta= os.path.join(DATADIR,categoria)
        valor=CATEGORIAS.index(categoria)
        # Parámetros de edición
        brillo = 1.15  # Ajusta el brillo (1.0 = sin cambios)
        contraste = 1.2  # Ajusta el contraste (1.0 = sin cambios)
        altas_luces = 1.0  # Ajusta las altas luces (1.0 = sin cambios)
        sombras = 1.3 # Ajusta las sombras (1.0 = sin cambios)
        nitidez = 1.5  # Ajusta la nitidez (1.0 = sin cambios)

        #para barra de carga
        list=os.listdir(carpeta)

        for i in tqdm(range(len(list)), desc=categoria):
            nombre_imagen= list[i]
            if nombre_imagen.endswith('.jpg') or nombre_imagen.endswith('.png') or nombre_imagen.endswith('.jpeg'):
                ruta = os.path.join(carpeta, nombre_imagen)
                imagen = Image.open(ruta)
                #plt.imshow(imagen)
                #plt.show()

                # Ajusta de parametros de la imagen
                ajuste_brillo = ImageEnhance.Brightness(imagen)
                imagen = ajuste_brillo.enhance(brillo)
                
                ajuste_contraste = ImageEnhance.Contrast(imagen)
                imagen = ajuste_contraste.enhance(contraste)

                ajuste_altasluces = ImageEnhance.Brightness(imagen)
                imagen = ajuste_altasluces.enhance(altas_luces)

                ajuste_sombra = ImageEnhance.Brightness(imagen)
                imagen = ajuste_sombra.enhance(sombras)

                ajuste_nitidez = ImageEnhance.Sharpness(imagen)
                imagen = ajuste_nitidez.enhance(nitidez)
                #plt.imshow(imagen)
                #plt.show()

                
                # Convertir a formato OpenCV y escalar
                imagen_cv = cv2.resize(np.array(imagen), (tamaño_img, tamaño_img))

                # Mostrar la imagen con el filtro de Sobel
                #plt.imshow(imagen_cv)
                #plt.show()

    rn.shuffle(data)
    x=[]
    y=[]
    lista=data
    for i in tqdm(range(len(lista)), desc="procesando"):
        par=data[i]
        x.append(par[0]) 
        y.append(par[1])
    x=np.array(x).reshape(-1,tamaño_img,tamaño_img,1)

    pickle_out=open("x.pickle","wb")
    pickle.dump(x,pickle_out)
    pickle_out.close()

    pickle_out=open("y.pickle","wb")
    pickle.dump(y,pickle_out)
    pickle_out.close()
    
    
generar_datos()