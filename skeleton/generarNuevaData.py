import cv2
import lecturaData as lec

import matplotlib.pyplot as plt
import numpy as np
import os

def generarRecorteMano(x, y, radio, imgInput):
    img = np.zeros((radio*2, radio*2, 3), dtype=np.uint8)

    
    if x!=0 and y!=0:

        yDesde = y-radio
        yHasta = y+radio
        xDesde = x-radio
        xHasta = x+radio

        if yDesde<0:
            yDesde = 0
        if yHasta>=imgInput.shape[0]:
            yHasta = imgInput.shape[0]
        if xDesde<0:
            xDesde = 0
        if xHasta>=imgInput.shape[1]:
            xHasta = imgInput.shape[1]

        img = imgInput[yDesde:yHasta, xDesde:xHasta]
        
    
        #Abajo
        if imgInput.shape[0]<=y+radio:
            aumentar = np.zeros((y+radio-imgInput.shape[0], img.shape[1], 3 ), dtype=np.uint8)
            img = np.concatenate((img, aumentar ), axis=0)

        #Arriba
        if y-radio<0:
            aumentar = np.zeros(( abs(y-radio), img.shape[1], 3 ), dtype=np.uint8)
            img = np.concatenate((aumentar, img), axis=0)

        #Derecha
        if imgInput.shape[1]<=x+radio:
            aumentar = np.zeros((img.shape[0], x+radio-imgInput.shape[1], 3 ), dtype=np.uint8)
            img = np.concatenate((img, aumentar ), axis=1)

        #Izquierda
        if x-radio<0:
            aumentar = np.zeros((img.shape[0], abs(x-radio), 3 ), dtype=np.uint8)
            img = np.concatenate((aumentar, img), axis=1)


    return img

def pintarSkeleton(imgInput, skeleton):

    img = np.copy(imgInput)

    for i in range(len(skeleton)):
        x = (int)(skeleton[i][0])
        y = (int)(skeleton[i][1])


        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

        if i == 0 or i == 1 :
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        if i == 2 or i == 5 :
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        if i == 3 or i == 6 :
            cv2.circle(img, (x, y), 5, (255, 0, 255), -1)

        if i == 4 or i == 7 : #manos
            cv2.circle(img, (x, y), 7, (255, 0, 0), -1)


    return img


def mostrarSkeleton(imgRGB, img, img1, img2, posFrame):



    plt.subplot(1, 4, 1)
    b,g,r = cv2.split(imgRGB)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)    
    plt.subplot(1, 4, 2)
    plt.imshow(img)
    plt.title( 'Figura ' + str(posFrame+1))

    plt.subplot(1, 4, 3)
    plt.imshow(img1)

    plt.subplot(1, 4, 4)
    plt.imshow(img2)

    plt.show()



def generarNuevosVideos(radio, vidRGB, vidDepth, videoSkeletos, mostrar):
    
    roi_vidRGB1 = []
    roi_vidRGB2 = []
    roi_vidDepth1 = []
    roi_vidDepth2 = []

    cantFrames = len(videoSkeletos)
    cantFrames_mitad = (int)(cantFrames/2)

    for posFrame in range(cantFrames):

        imgRGB = vidRGB[posFrame]
        imgDepth = vidDepth[posFrame]
        frame_skeleto = videoSkeletos[posFrame]

        x = (int)(frame_skeleto[4][0])
        y = (int)(frame_skeleto[4][1])
        roi_imgRGB1 = generarRecorteMano(x, y, radio, imgRGB)
        roi_imgDepth1 = generarRecorteMano(x, y, radio, imgDepth)
        
        x = (int)(frame_skeleto[7][0])
        y = (int)(frame_skeleto[7][1])
        roi_imgRGB2 = generarRecorteMano(x, y, radio, imgRGB)
        roi_imgDepth2 = generarRecorteMano(x, y, radio, imgDepth)

        if mostrar and posFrame<cantFrames_mitad:
            img = pintarSkeleton(imgDepth, frame_skeleto)
            mostrarSkeleton(imgRGB, img, roi_imgRGB1, roi_imgRGB2, posFrame)

        roi_vidRGB1.append(roi_imgRGB1)
        roi_vidRGB2.append(roi_imgRGB2)
        roi_vidDepth1.append(roi_imgDepth1)
        roi_vidDepth2.append(roi_imgDepth2)

    roi_vidRGB1 = np.array(roi_vidRGB1)
    roi_vidRGB2 = np.array(roi_vidRGB2)
    roi_vidDepth1 = np.array(roi_vidDepth1)
    roi_vidDepth2 = np.array(roi_vidDepth2)



    return roi_vidRGB1, roi_vidRGB2, roi_vidDepth1, roi_vidDepth2




'''
cantData = 100
pathsRGB, pathsDepth, clases = lec.leerNombres('../train_list.txt', 
    "../DataOriginal/", cantData, numEtiquetas=100)

pathsSkeleton, _, _ = lec.leerNombres('../train_list.txt', 
    "../data_skeleton/", cantData, numEtiquetas=100)
pathsSkeleton = lec.cambiarExtension(pathsSkeleton, "_Skel.mat")


numVid = 3


vidRGB = lec.leer_una_sena_video_intercalar(False, 64, 64, pathsRGB[numVid])
vidRGB = np.array(vidRGB)
vidDepth = lec.leer_una_sena_video_intercalar(False, 64, 64, pathsDepth[numVid])
vidDepth = np.array(vidDepth)
videoSkeletos = lec.leerSkeleton(pathsSkeleton[numVid])

print(vidRGB.shape)
print(vidDepth.shape)
print(len(videoSkeletos))
print(pathsSkeleton[numVid])


radio = 60

roi_vidRGB1, roi_vidRGB2, roi_vidDepth1, roi_vidDepth2 = generarNuevosVideos(radio, vidRGB, 
    vidDepth, videoSkeletos, True)

'''


def simularSkeleton(cantidad):
    frames_puntos = []

    for _ in range(cantidad):
        frame_puntos = np.zeros((8,2))
        frames_puntos.append( frame_puntos )

    return frames_puntos
  

def generarSilent_Data(cantData, grupoData, radio):

    numEtiquetas = 300

    pathsRGB, pathsDepth, clases = lec.leerNombres(grupoData, 
        "../DataOriginal/", cantData, numEtiquetas=numEtiquetas)

    pathsSkeleton, _, _ = lec.leerNombres(grupoData, 
        "../data_skeleton/", cantData, numEtiquetas=numEtiquetas)
    pathsSkeleton = lec.cambiarExtension(pathsSkeleton, "_Skel.mat")


    pathsRGB_W, pathsDepth_W, _ = lec.leerNombres(grupoData, 
        "../DataSilent2/", cantData, numEtiquetas=numEtiquetas)


    cantFramesPobres = 0
    cantFramesRGBNormal = 0
    cantFramesSkeletoPobre = 0

    for i in range(cantData):
        # 61, 94
        #i=4929+i

        #print(pathsRGB[i])

        vidRGB = lec.leer_una_sena_video_intercalar(False, 64, 64, pathsRGB[i])
        vidDepth = lec.leer_una_sena_video_intercalar(False, 64, 64, pathsDepth[i])

        try:
            videoSkeletos = lec.leerSkeleton(pathsSkeleton[i])
        except:
            print("Error", pathsRGB[i])
            videoSkeletos = simularSkeleton(len(vidRGB))
        
        if (grupoData=='../test_list.txt') and (i==850 or i==2644 or i==2908 or i==4929):#2900
        	videoSkeletos = simularSkeleton(len(vidRGB))

        if len(videoSkeletos)>len(vidRGB):
            vidRGB = lec.leer_una_sena_video(False, 64, 64, pathsRGB[i])
            vidDepth = lec.leer_una_sena_video(False, 64, 64, pathsDepth[i])
            cantFramesRGBNormal = cantFramesRGBNormal + 1
        
        if len(videoSkeletos)<len(vidRGB):
            vidRGB = lec.leer_una_sena_video_intercalar2(False, 64, 64, pathsRGB[i])
            vidDepth = lec.leer_una_sena_video_intercalar2(False, 64, 64, pathsDepth[i])
            cantFramesSkeletoPobre = cantFramesSkeletoPobre + 1
        

        '''
        print(pathsRGB[i])
        print(len(vidRGB))
        print(len(vidDepth))
        print(len(videoSkeletos))
        '''


        roi_vidRGB1, roi_vidRGB2, roi_vidDepth1, roi_vidDepth2 = generarNuevosVideos(radio, vidRGB, 
        vidDepth, videoSkeletos, False)



        directorioPadre = pathsRGB_W[i][0: (len(pathsRGB_W[i])-16) ]
        directorioHijo = pathsRGB_W[i][len(pathsRGB_W[i])-15:(len(pathsRGB_W[i])-12)]
        
        nameRGB = pathsRGB_W[i][(len(pathsRGB_W[i])-11): (len(pathsRGB_W[i])-4)]
        nameDepth = pathsDepth_W[i][(len(pathsDepth_W[i])-11): (len(pathsDepth_W[i])-4)]
        
        
        try:
            os.stat(directorioPadre)
        except:
            os.mkdir(directorioPadre)
        
        try:
            os.stat(directorioPadre+"/"+directorioHijo)
        except:
            os.mkdir(directorioPadre+"/"+directorioHijo)


        dirPathRGB_W = directorioPadre+"/"+directorioHijo+"/"+nameRGB+".avi"
        dirPathDepth_W = directorioPadre+"/"+directorioHijo+"/"+nameDepth+".avi"

        dirPathRGB_W_R1 = directorioPadre+"/"+directorioHijo+"/"+nameRGB+"_R1.avi"
        dirPathRGB_W_R2 = directorioPadre+"/"+directorioHijo+"/"+nameRGB+"_R2.avi"

        dirPathDepth_W_R1 = directorioPadre+"/"+directorioHijo+"/"+nameDepth+"_R1.avi"
        dirPathDepth_W_R2 = directorioPadre+"/"+directorioHijo+"/"+nameDepth+"_R2.avi"

        dirPathDepth_Skeleto = directorioPadre+"/"+directorioHijo+"/"+nameRGB+"_S.npy"
        
        
        ancho = len(vidRGB[0][0])
        alto = len(vidRGB[0])
        outRGB = cv2.VideoWriter(dirPathRGB_W,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (ancho,alto))
        outDepth = cv2.VideoWriter(dirPathDepth_W,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (ancho,alto))

        outRGB_R1 = cv2.VideoWriter(dirPathRGB_W_R1,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (radio*2, radio*2))
        outRGB_R2 = cv2.VideoWriter(dirPathRGB_W_R2,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (radio*2, radio*2))
        outDepth_R1 = cv2.VideoWriter(dirPathDepth_W_R1,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (radio*2, radio*2))
        outDepth_R2 = cv2.VideoWriter(dirPathDepth_W_R2,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (radio*2, radio*2))

        outSkeletos = []


        last_frame_RGB = cv2.cvtColor(vidRGB[0], cv2.COLOR_BGR2GRAY)
        
        outRGB.write(vidRGB[0])
        outDepth.write(vidDepth[0])    

        outRGB_R1.write(roi_vidRGB1[0])
        outRGB_R2.write(roi_vidRGB2[0])
        outDepth_R1.write(roi_vidDepth1[0])
        outDepth_R2.write(roi_vidDepth2[0])

        outSkeletos.append(videoSkeletos[0])



        cantFrames = 1
        for j in range(len(vidRGB)):
            current_frame_RGB = cv2.cvtColor(vidRGB[j], cv2.COLOR_BGR2GRAY)
            
            # Find the absolute difference between frames
            diff = cv2.absdiff(last_frame_RGB, current_frame_RGB)
            # If difference is greater than a threshold, that means motion detected.
            if np.mean(diff) > 2:            
                outRGB.write(vidRGB[j])
                outDepth.write(vidDepth[j])

                outRGB_R1.write(roi_vidRGB1[j])
                outRGB_R2.write(roi_vidRGB2[j])
                outDepth_R1.write(roi_vidDepth1[j])
                outDepth_R2.write(roi_vidDepth2[j])

                outSkeletos.append(videoSkeletos[j])
                
                cantFrames = cantFrames + 1
                
            last_frame_RGB = current_frame_RGB
        
        outRGB.release()
        outDepth.release()

        outRGB_R1.release()
        outRGB_R2.release()
        outDepth_R1.release()
        outDepth_R2.release()




        
        
        
        
        if (cantFrames<3):
            cantFramesPobres = cantFramesPobres + 1
            print(dirPathRGB_W)
            outRGB2 = cv2.VideoWriter(dirPathRGB_W,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (ancho,alto))
            outDepth2 = cv2.VideoWriter(dirPathDepth_W,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (ancho,alto))

            outRGB_R1_2 = cv2.VideoWriter(dirPathRGB_W_R1,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (radio*2, radio*2))
            outRGB_R2_2 = cv2.VideoWriter(dirPathRGB_W_R2,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (radio*2, radio*2))
            outDepth_R1_2 = cv2.VideoWriter(dirPathDepth_W_R1,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (radio*2, radio*2))
            outDepth_R2_2 = cv2.VideoWriter(dirPathDepth_W_R2,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (radio*2, radio*2))

            outSkeletos = []

            last_frame_RGB = cv2.cvtColor(vidRGB[0], cv2.COLOR_BGR2GRAY)

            outRGB2.write(vidRGB[0])
            outDepth2.write(vidDepth[0])

            outRGB_R1_2.write(roi_vidRGB1[0])
            outRGB_R2_2.write(roi_vidRGB2[0])
            outDepth_R1_2.write(roi_vidDepth1[0])
            outDepth_R2_2.write(roi_vidDepth2[0])

            outSkeletos.append(videoSkeletos[0])

            for j in range(len(vidRGB)):
                current_frame_RGB = cv2.cvtColor(vidRGB[j], cv2.COLOR_BGR2GRAY)

                # Find the absolute difference between frames
                diff = cv2.absdiff(last_frame_RGB, current_frame_RGB)
                # If difference is greater than a threshold, that means motion detected.
                if np.mean(diff) > 0.7:
                    outRGB2.write(vidRGB[j])
                    outDepth2.write(vidDepth[j])

                    outRGB_R1_2.write(roi_vidRGB1[j])
                    outRGB_R2_2.write(roi_vidRGB2[j])
                    outDepth_R1_2.write(roi_vidDepth1[j])
                    outDepth_R2_2.write(roi_vidDepth2[j])

                    outSkeletos.append(videoSkeletos[j])

                last_frame_RGB = current_frame_RGB

            outRGB2.release()
            outDepth2.release()

            outRGB_R1_2.release()
            outRGB_R2_2.release()
            outDepth_R1_2.release()
            outDepth_R2_2.release()


        np.save(dirPathDepth_Skeleto, outSkeletos)
            
        if i%100 == 0 :
            print("Videos Leidos: ", i+1 )
        #print("Videos Leidos: ", i+1 )
        
        
        #break

    print("Frames Pobres ", cantFramesPobres)
    print("cantFramesRGBNormal ", cantFramesRGBNormal)
    print("cantFramesSkeletoPobre ", cantFramesSkeletoPobre)




radio = 60


cantData = 35878
grupoData = '../train_list.txt'
#generarSilent_Data(cantData, grupoData, radio)

cantData = 6271
grupoData = '../test_list.txt'
generarSilent_Data(cantData, grupoData, radio)



