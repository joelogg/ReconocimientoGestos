import lecturaData as lec


cantData = 200
pathNombresRGB, pathNombresDepth, clases = lec.leerNombres('../valid_list.txt', 
    "SilentsData/NoSilent_", cantData, numEtiquetas=100)
print(len(pathNombresRGB))
print(len(pathNombresDepth))
print(len(clases))






import cv2
import numpy as np
import glob

#Se lee los videos rgb
def leer_una_sena_video(nombreVid):
    
    cap = cv2.VideoCapture('../'+nombreVid)
    vid = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        #vid.append(  cv2.resize(img, (160, 120) )   )    
        vid.append(  cv2.resize(img, (64,64))   )    
        #vid.append(img)    
    return vid




def recortarVideo(vid, tam):
    razon = 1.0*len(vid)/tam
    vidR = []
    if (len(vid)<=0):
        return vidR
    for i in range(tam):
        vidR.append( vid[(int)(i*razon)] )
    del vid
    return vidR



def formatearVideo(vid):
    vid = np.array(vid)
    return vid


def juntarRGBandDepth(rgb, depth):
    return np.array( np.concatenate((rgb, depth)) )


def espejoVideo(vid):
    vidNew = []
    for img in vid:
        vidNew.append( cv2.flip( img, 1 ) )
    del vid
    return vidNew


def combinar(vid):
    vidRes = []
    vidRes.append(vid[0])
    vidRes.append(vid[3])
    vidRes.append(vid[1])
    vidRes.append(vid[4])
    vidRes.append(vid[2])
    vidRes.append(vid[5])
    return np.array(vidRes)


from skimage import exposure

def normalization_min_max(vid):
    vidNew = []
    
    for img in vid:
        img_norm = (img - np.min(img))/ (np.max(img) - np.min(img))
        #img_norm = exposure.equalize_adapthist(img, clip_limit=0.03)
        img_norm = exposure.equalize_hist(img_norm)

        
        vidNew.append( img_norm )
    del vid
    return vidNew


import numpy as np
import cv2

def substraerFondo(vidEntrada):
    vid = vidEntrada.copy()
    
    vidNew = []
    first_frame = vid[0]
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
    for frame in vid:

        frameAux = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameAux = cv2.GaussianBlur(frameAux, (5, 5), 0)
        frameAux = cv2.absdiff(first_gray, frameAux)

        _, frameAux = cv2.threshold(frameAux, 20, 255, cv2.THRESH_BINARY)
        
        frame[ frameAux==0 ] = 0
        vidNew.append(frame)
    
    del vid
    return np.array(vidNew)






numVid = 4
vidRGB = leer_una_sena_video(pathNombresRGB[numVid])
vidRGB = recortarVideo(vidRGB, 16)   
vidRGB = np.array(vidRGB)
print(vidRGB.shape)



vidDepth = leer_una_sena_video(pathNombresDepth[numVid])
vidDepth = recortarVideo(vidDepth, 16)   
vidDepth = np.array(vidDepth)
print(vidDepth.shape)




vidF = substraerFondo(vidRGB)
vidF.shape
print(vidF.shape)


import matplotlib.pyplot as plt

fig = plt.figure()


for i in range(1):
    plt.subplot(1, 3, 1)
    b,g,r = cv2.split(vidRGB[i])       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)
    
    plt.subplot(1, 3, 2)
    b,g,r = cv2.split(vidF[i])       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)
    
    plt.subplot(1, 3, 3)
    plt.imshow(vidDepth[i])
    
    plt.show()



'''
cantFrames = 16
bach_size = 8

cantData = 150
pathNombresRGB, pathNombresDepth, clases = leerNombresTrain('../train_list.txt', cantData)
len(pathNombresRGB)


from keras import utils
dataY_Entrada = utils.to_categorical(clases)
dataY_Entrada.shape



def my_generator(pathNombresRGB, pathNombresDepth, cantFrames, cantData, dataY_Entrada, bach_size):
    i = 0
    boolEspejo = False
    while True:
        
        
        
        dataX = []
        dataXX = []
        dataY = []
        
        for _ in range(bach_size):
            
            if(i >= cantData*2):
            #if(i >= cantData):
                i=0
            
            if (boolEspejo):
                vidRGB = leer_una_sena_video(pathNombresRGB[(int)(i/2)])        
                vidRGB = recortarVideo(vidRGB, cantFrames)
                vidRGB = np.array(vidRGB)
                
                #vidDepth = leer_una_sena_video(pathNombresDepth[(int)(i/2)])
                #vidDepth = recortarVideo(vidDepth, cantFrames)
                #vidDepth = np.array(vidDepth)
                
                
                #vidRGB = substraerFondo(vidRGB)
                vidRGB = espejoVideo(vidRGB)
                vidRGB = formatearVideo(vidRGB)
                
                #vidDepth = espejoVideo(vidDepth)
                #vidDepth = formatearVideo(vidDepth)


                boolEspejo = False

            else:
                vidRGB = leer_una_sena_video(pathNombresRGB[(int)(i/2)])        
                vidRGB = recortarVideo(vidRGB, cantFrames)
                vidRGB = np.array(vidRGB)
                
                #vidDepth = leer_una_sena_video(pathNombresDepth[(int)(i/2)])
                #vidDepth = recortarVideo(vidDepth, cantFrames)
                #vidDepth = np.array(vidDepth)
                
                #vidRGB = substraerFondo(vidRGB)
                #vidRGB = formatearVideo(vidRGB)          

                boolEspejo = True
                
            
            dataX.append(vidRGB)
            #dataXX.append(vidDepth)            
            
            valY = dataY_Entrada[(int)(i/2)]
            dataY.append(valY)
            
            i += 1
        
        
        dataX = np.array(dataX)
        #dataXX = np.array(dataXX)
        
        #yield [dataX, dataXX], np.array(dataY)
        yield dataX, np.array(dataY)





import numpy as np

bach_size_T = 64
bach_size_V = bach_size_T
cantFrames = 16

cantData_T = 35878
pathNombresRGB_T, pathNombresDepth_T, clases_T = leerNombresTrain('../train_list.txt', cantData_T)
print(len(pathNombresRGB_T))
dataY_Entrada_T = utils.to_categorical(clases_T)
print(dataY_Entrada_T.shape)

cantData_V = 5784
pathNombresRGB_V, pathNombresDepth_V, clases_V = leerNombresTrain('../valid_list.txt', cantData_V)
print(len(pathNombresRGB_V))
dataY_Entrada_V = utils.to_categorical(clases_V)
print(dataY_Entrada_V.shape)





cantData_T = len(pathNombresRGB_T)
cantData_V = len(pathNombresRGB_V)
cantData_T, cantData_V


# Generators
training_generator = my_generator(pathNombresRGB_T, pathNombresDepth_T, cantFrames, cantData_T, dataY_Entrada_T, bach_size_T)
validation_generator = my_generator(pathNombresRGB_V, pathNombresDepth_V, cantFrames, cantData_V, dataY_Entrada_V, bach_size_V)




from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling3D, ZeroPadding3D, Conv3D
from keras.optimizers import Adam
from keras.layers import LSTM
from keras.models import Model
from keras.layers import Input, Activation, Embedding, merge, LSTM, Dropout, Dense, RepeatVector, BatchNormalization, \
    TimeDistributed, Flatten, Reshape, concatenate
from keras.layers.convolutional_recurrent import ConvLSTM2D



def get_model(summary=False):
    """ Return the Keras model of the network
    """
    
    inputs_RGB = Input(shape=(16, 112, 112, 3))
    
    # 1st layer group
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv1', strides=(1, 1, 1))(inputs_RGB)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)
    
    # 2nd layer group
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
    
    # 3rd layer group
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a', strides=(1, 1, 1))(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)
    
    # 4th layer group
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a', strides=(1, 1, 1))(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b', strides=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)
    
    # 5th layer group
    x = Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='conv5a', strides=(1, 1, 1))(x)
    x = Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='conv5b', strides=(1, 1, 1))(x)
    x = ZeroPadding3D(padding=(0, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(x)
    x = Flatten()(x)
    
    
    inputs_Depth = Input(shape=(16, 112, 112, 3))
    
    # 1st layer group
    y = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='Dconv1', strides=(1, 1, 1))(inputs_Depth)
    y = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='Dpool1')(y)
    
    # 2nd layer group
    y = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='Dconv2', strides=(1, 1, 1))(y)
    y = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='Dpool2')(y)
    
    # 3rd layer group
    y = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='Dconv3a', strides=(1, 1, 1))(y)
    y = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='Dconv3b', strides=(1, 1, 1))(y)
    y = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='Dpool3')(y)
    
    # 4th layer group
    y = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='Dconv4a', strides=(1, 1, 1))(y)
    y = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='Dconv4b', strides=(1, 1, 1))(y)
    y = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='Dpool4')(y)
    
    # 5th layer group
    y = Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='Dconv5a', strides=(1, 1, 1))(y)
    y = Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='Dconv5b', strides=(1, 1, 1))(y)
    y = ZeroPadding3D(padding=(0, 1, 1))(y)
    y = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='Dpool5')(y)
    y = Flatten()(y)
        
    
   
    merged = concatenate([x, y])
    
    
    # FC layers group
    x = Dense(4096, activation='relu', name='fc6')(merged)
    x = Dropout(.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(.5)(x)
    predictions = Dense(100, activation='softmax', name='fc8')(x)
    
    
    model = Model(inputs=[inputs_RGB, inputs_Depth], outputs=predictions)
    
    if summary:
        print(model.summary())
    return model


model = get_model(summary=True)


import keras
from keras.optimizers import Adam, SGD

sgd = SGD(lr=0.001, decay=5e-5, momentum=0.9)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['acc'])





from keras.callbacks import TensorBoard

cantDataDoble_T = cantData_T*2
cantDataDoble_V = cantData_V*2

#cantDataDoble_T = cantData_T
#cantDataDoble_V = cantData_V
print(bach_size_T)
print(bach_size_V)


result_train = model.fit_generator(
    generator=training_generator,
    epochs=20,
    steps_per_epoch=cantDataDoble_T // bach_size_T,
    validation_data=validation_generator,
    validation_steps=cantDataDoble_V // bach_size_V)


#result_train = model.fit_generator(
#    generator=training_generator,
#    epochs=20,
#    steps_per_epoch=cantDataDoble_T // bach_size_T)




import matplotlib.pyplot as plt
accuracy = result_train.history['acc']
val_accuracy = result_train.history['val_acc']
loss = result_train.history['loss']
val_loss = result_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validat1137153748ion accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
plt.show()


import h5py

model.save_weights('t_weights_RGBD_4.h5', overwrite=True)
json_string = model.to_json()
with open('t_model_RGBD_4.json', 'w') as f:
    f.write(json_string)




scores = model.evaluate_generator(generator=training_generator, steps=cantDataDoble_T // bach_size_T, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


scores = model.evaluate_generator(generator=validation_generator, steps=cantDataDoble_V // bach_size_V, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



bach_size_Test = bach_size_T
cantFrames = 16

cantData_Test = 6271
pathNombresRGB_Test, pathNombresDepth_Test, clases_Test = leerNombresTrain('../test_list.txt', cantData_Test)
print(len(pathNombresRGB_Test))
dataY_Entrada_Test = utils.to_categorical(clases_Test)
print(dataY_Entrada_Test.shape)

# Generators
test_generator = my_generator(pathNombresRGB_Test, pathNombresDepth_Test, cantFrames, cantData_Test, dataY_Entrada_Test, bach_size_Test)
cantData_Test = len(pathNombresRGB_Test)




scores = model.evaluate_generator(generator=test_generator, steps=cantData_Test // bach_size_Test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


'''