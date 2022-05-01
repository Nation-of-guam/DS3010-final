import os
import pandas as pd
import numpy as np
import matplotlib
import seaborn
import cv2
import os
import glob
from bs4 import BeautifulSoup as bs
from lxml import etree
import sklearn as sk
from sklearn.model_selection import train_test_split
import keras
import tensorflow
from keras.layers import *


def resize(f, size):
    t = etree.parse(f)
    for dimmensions in t.xpath("size"):
        x = int(dimmensions.xpath("width")[0].text)
        y = int(dimmensions.xpath("width")[0].text)
    for dimmensions in t.xpath("object/bndbox"):
        x_min = int(dimmensions.xpath("xmin")[0].text) / (x / size)
        y_min = int(dimmensions.xpath("ymin")[0].text) / (y / size)
        x_max = int(dimmensions.xpath("xmax")[0].text) / (x / size)
        y_max = int(dimmensions.xpath("ymax")[0].text) / (y / size)
    return [x_max,y_max,x_min,y_min]


def load(size):
    image_path = os.path.join('archive/images', '*g')
    image_files = glob.glob(image_path)
    image_files.sort()
    X = []
    for f1 in image_files:
        image = cv2.imread(f1)
        image = cv2.resize(image, (size, size))
        X.append(np.array(image))
    Y = []
    files_Y = []
    for i in range(433):
         files_Y.append(os.path.join('archive/annotations/Cars'+str(i)+'.xml'))
    for i in files_Y:
        Y.append(resize(i, size))
    return X,Y

size = 256
X,Y = load(size)

print((419)/(500/256))
print(Y[0])

X = (np.array(X))/255
Y = (np.array(Y))/255

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

license_model = keras.models.Sequential()
license_model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(255, 255, 3)))
license_model.add(keras.layers.MaxPooling2D((2, 2)))
license_model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
license_model.add(keras.layers.MaxPooling2D((2, 2)))
license_model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
license_model.add(Flatten())
license_model.add(Dense(64, activation="relu"))
license_model.add(Dense(4, activation="sigmoid"))

license_model.compile(optimizer='SGD',
              loss='mean_squared_error',
             metrics=['accuracy'])

license_model.summary()

model_train = license_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=64, verbose=1)

scores = license_model.evaluate(X_test, y_test, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))




