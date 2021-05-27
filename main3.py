import pickle
import cv2
import os
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import optimizers
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil


# for file in os.listdir("./data1/data/train"):
#     if "cat" in file:
#         shutil.move(f"./data1/data/train/{file}","./data1/data/train/cat/")
#     else:
#         shutil.move(f"./data1/data/train/{file}","./data1/data/train/dog/")

# train_path = "./data1/data/train"
test_path = "./data1/data/test1"
# training_data = ImageDataGenerator(rescale=1/255,horizontal_flip=True,rotation_range=90,brightness_range=[0.2,1.0],zoom_range=[0.5,1.0]).flow_from_directory(directory=train_path,target_size=(64,64), class_mode='categorical', batch_size=50,shuffle=True)
testing_data = ImageDataGenerator(rescale=1/255,horizontal_flip=True,rotation_range=90,brightness_range=[0.2,1.0],zoom_range=[0.5,1.0]).flow_from_directory(directory=test_path,target_size=(64,64), class_mode='categorical', batch_size=50, shuffle=True)

# model = keras.Sequential([
#     keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding="same", input_shape=(64,64,3),activation='relu'),
#     keras.layers.MaxPool2D((2,2)),
#     keras.layers.Conv2D(filters=128, kernel_size=(3,3),padding="same", activation="relu"),
#     keras.layers.MaxPool2D((2,2)),
#     keras.layers.Conv2D(filters=256, kernel_size=(3,3),padding="same", activation="relu"),
#     keras.layers.MaxPool2D((2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(512,activation="relu"),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(2,activation="softmax")
# ])

# model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=["accuracy"])

# print(model.summary())
# model.fit_generator(training_data,epochs=15)

# model.save("model3.h5")

model = keras.models.load_model("model3.h5")
# img_array = np.array(cv2.resize(cv2.imread("test1.jpg"),(200,200))).flatten()

# print(model.predict(img_array.reshape(1,200,200,3)))
model.evaluate(testing_data)