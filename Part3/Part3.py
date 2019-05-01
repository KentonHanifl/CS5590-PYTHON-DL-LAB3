import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import os
from keras.models import model_from_json
from scipy.ndimage import imread
from sklearn.model_selection import train_test_split

#for cropping the images to make them the same size input
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

#metaparameters
epochsNum = 50
lrate = 0.01
sgd = SGD(lr=lrate)

#load dataset
datasetPath = "natural_images"
xcrop = 40
ycrop = 40

Ims = np.ndarray([6899,xcrop,ycrop,1])
Labels = np.ndarray([6899,1])
classes = os.listdir(datasetPath)
num_classes = len(classes)

print("starting loading (takes about 15 seconds)")
imnum = 0
for classnum,c in enumerate(classes):
    classIms = os.listdir(datasetPath+"/"+c)
    classImNum = len(classIms)
    for im in classIms:
        uncroppedIm = imread(datasetPath+"/"+c+"/"+im,flatten=True)
        Ims[imnum,:,:,:] = crop_center(uncroppedIm,xcrop,ycrop).reshape([xcrop,ycrop,1])
        Labels[imnum,:] = classnum
        imnum += 1
        


X_train, X_test, Y_train, Y_test = train_test_split(Ims, Labels,
                                                    test_size=0.25, random_state=87)

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

print("done loading dataset")

#load and compile model
try:
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    input("press enter to exit")
except:
    #Create model
    model = Sequential()
    model.add(Conv2D(xcrop, (3, 3), input_shape=(xcrop, ycrop,1), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), activation='relu',padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    #fit model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochsNum, batch_size=200, verbose = 1)

    #save model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

    #evaluate model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    input("press enter to exit")
