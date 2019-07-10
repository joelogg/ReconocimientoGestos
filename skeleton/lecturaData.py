import re

#Leer los nombres de la data (RGB, profundida, tipo)
def leerNombres(path, prePath, cantMaxima, numEtiquetas):
    f = open(path, 'r+')
    
    pathsRGB = []
    pathsDepth = []
    clases = []
    cant = 0
    for line in f.readlines():
        
        line = line.split(" ")
        
        label = int( re.sub("\D", "", line[2]) ) -1
        if label<numEtiquetas:
            pathsRGB.append(prePath+line[0])
            pathsDepth.append(prePath+line[1])
            clases.append( label )

            cant = cant + 1
            if cant>=cantMaxima:
                break
        
    f.close()
    
    return pathsRGB, pathsDepth, clases

'''
print("--------Leer nombres RGB y Depth--------")
cantData = 100
pathsRGB, pathsDepth, clases = leerNombres('../train_list.txt', 
    "../DataOriginal/", cantData, numEtiquetas=100)
print(len(pathsRGB))
print(len(pathsDepth))
print(len(clases))

print(pathsRGB[0])
print(pathsDepth[0])
print(clases[0])
print("-------------------------------------")
'''



import cv2

#Se lee los videos rgb
def leer_una_sena_video(bool_resize, ancho, alto, nombreVid):
    
    cap = cv2.VideoCapture(nombreVid)
    vid = []

    while True:
        ret, img = cap.read()
        if not ret:
            break

        if bool_resize:
            vid.append(  cv2.resize(img, (64,64))   )    
        else:
            vid.append(img)

    return vid


def leer_una_sena_video_intercalar(bool_resize, ancho, alto, nombreVid):
    
    cap = cv2.VideoCapture(nombreVid)
    vid = []
    intercalar = True

    while True:
        
        ret, img = cap.read()
        if not ret:
            break

        if intercalar:
            if bool_resize:
                vid.append(  cv2.resize(img, (64,64))   )    
            else:
                vid.append(img)

            intercalar = False

        else:
            intercalar = True

    return vid

def leer_una_sena_video_intercalar2(bool_resize, ancho, alto, nombreVid):
    
    cap = cv2.VideoCapture(nombreVid)
    vid = []
    intercalar = 0

    while True:
        
        ret, img = cap.read()
        if not ret:
            break

        if intercalar%3==0:
            if bool_resize:
                vid.append(  cv2.resize(img, (64,64))   )    
            else:
                vid.append(img)

        intercalar = intercalar+1

    return vid


'''
print("--------Leer videos RGB y Depth--------")

import numpy as np

cantData = 100
pathsRGB, pathsDepth, clases = leerNombres('../train_list.txt', 
    "../DataOriginal/", cantData, numEtiquetas=100)
vidRGB = leer_una_sena_video_intercalar(False, 64, 64, pathsRGB[0])
vidRGB = np.array(vidRGB)
print(vidRGB.shape)
vidDepth = leer_una_sena_video_intercalar(False, 64, 64, pathsDepth[0])
vidDepth = np.array(vidDepth)
print(vidDepth.shape)



import matplotlib.pyplot as plt
pos = (int)(len(vidRGB)/2)
plt.subplot(1, 2, 1)
b,g,r = cv2.split(vidRGB[pos])       # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb
plt.imshow(rgb_img)    
plt.subplot(1, 2, 2)
plt.imshow(vidDepth[pos])
plt.show()


print("-------------------------------------")
'''



import scipy.io
import numpy as np

def cambiarExtension(pathsSkeleton, newExtension):
    for i in range(len(pathsSkeleton)):
        pathsSkeleton[i] = pathsSkeleton[i][:-4] + newExtension
    return pathsSkeleton


def leerSkeleton(pathSkeleto ):
    frames_puntos = []

    data = scipy.io.loadmat(pathSkeleto)['skeletonData']
    data = data[0]

    
    for puntos_frame in data:
        
        frame_puntos = np.zeros((8,2))

        if len(puntos_frame[0][0][1])>0 and len(puntos_frame[0][0][0])>0:         

            frame_puntos_ubi = puntos_frame[0][0][1][0]
            frame_puntos_ori = puntos_frame[0][0][0]
            
            
            for i in range(len(frame_puntos_ubi)):
                pos = (int)(frame_puntos_ubi[i])
                
                if pos != 0:
                    if i==0 : #Cabeza
                        frame_puntos[0][0] = frame_puntos_ori[pos-1][0]
                        frame_puntos[0][1] = frame_puntos_ori[pos-1][1]

                    if i==1: #Pecho
                        frame_puntos[1][0] = frame_puntos_ori[pos-1][0]
                        frame_puntos[1][1] = frame_puntos_ori[pos-1][1]

                    if i==2: #Hombro1
                        frame_puntos[2][0] = frame_puntos_ori[pos-1][0]
                        frame_puntos[2][1] = frame_puntos_ori[pos-1][1]

                    if i==3: #Codo1
                        frame_puntos[3][0] = frame_puntos_ori[pos-1][0]
                        frame_puntos[3][1] = frame_puntos_ori[pos-1][1]

                    if i==4: #Muñeca1
                        frame_puntos[4][0] = frame_puntos_ori[pos-1][0]
                        frame_puntos[4][1] = frame_puntos_ori[pos-1][1]

                    if i==5: #Hombro2
                        frame_puntos[5][0] = frame_puntos_ori[pos-1][0]
                        frame_puntos[5][1] = frame_puntos_ori[pos-1][1]

                    if i==6: #Codo2
                        frame_puntos[6][0] = frame_puntos_ori[pos-1][0]
                        frame_puntos[6][1] = frame_puntos_ori[pos-1][1]

                    if i==7: #Muñeca2
                        frame_puntos[7][0] = frame_puntos_ori[pos-1][0]
                        frame_puntos[7][1] = frame_puntos_ori[pos-1][1]
        

        frames_puntos.append( frame_puntos )

    return frames_puntos



'''
print("--------Leer Skeleton--------")

cantData = 100
pathsRGB, pathsDepth, clases = leerNombres('../train_list.txt', 
    "../DataOriginal/", cantData, numEtiquetas=100)


pathsSkeleton, _, _ = leerNombres('../train_list.txt', 
    "../data_skeleton/", cantData, numEtiquetas=100)
pathsSkeleton = cambiarExtension(pathsSkeleton, "_Skel.mat")


numVid = 10

vidRGB = leer_una_sena_video_intercalar(False, 64, 64, pathsRGB[numVid])
vidRGB = np.array(vidRGB)
vidDepth = leer_una_sena_video_intercalar(False, 64, 64, pathsDepth[numVid])
vidDepth = np.array(vidDepth)
frames_puntos = leerSkeleton(pathsSkeleton[numVid])



print(vidRGB.shape)
print(vidDepth.shape)
print(len(frames_puntos))
print(pathsSkeleton[numVid])



import matplotlib.pyplot as plt


for i in range(len(frames_puntos)):
    posFrame = i

    img = vidDepth[posFrame]
    frame_puntos = frames_puntos[posFrame]


    for i in range(len(frame_puntos)):
        x = (int)(frame_puntos[i][0])
        y = (int)(frame_puntos[i][1])


        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

        if i == 0 or i == 1 :
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        if i == 2 or i == 5 :
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        if i == 3 or i == 6 :
            cv2.circle(img, (x, y), 5, (255, 0, 255), -1)

        if i == 4 or i == 7 :
            cv2.circle(img, (x, y), 7, (255, 0, 0), -1)





    
    plt.subplot(1, 2, 1)
    b,g,r = cv2.split(vidRGB[posFrame])       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)    
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title( 'Figura ' + str(posFrame+1))
    plt.show()

'''