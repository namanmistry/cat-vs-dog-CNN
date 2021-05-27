import pickle
import cv2
import os
import numpy as np
import random
from tensorflow import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# data = []


# for file in os.listdir("./data/train"):
#     label = 0
#     if "cat" in file:
#         label = 1
#     else:
#         label = 0
#     img = cv2.imread(f"./data/train/{file}")
#     img = cv2.resize(img,(80,80))
#     img_array = np.array(img).flatten()
#     data.append([img_array,label])
# print(len(data))
# random.shuffle(data)
# test_data = data[20000:]
# data = data[:20000]
# print(len(data))
# print(len(test_data))
# pickle_open = open("data.pkl", "wb")
# pickle.dump(data,pickle_open)
# pickle_open.close()

# pickle_open = open("test.pkl", "wb")
# pickle.dump(test_data,pickle_open)
# pickle_open.close()

# pickle_open = open("data.pkl","rb")
# data = pickle.load(pickle_open)
# random.shuffle(data)
# count = 0
# features = []
# labels = []

# for feature, label in data:
#     features.append(feature)
#     labels.append(label)
#     if count>15000:
#         break
#     count+=1

# features = np.array(features)
# labels = np.array(labels)

# features = features/255

# print(features)

# pickle_open = open("feature.pkl","wb")
# pickle.dump(features,pickle_open)

# pickle_open = open("labels.pkl","wb")
# pickle.dump(labels,pickle_open)

# print(len(labels))
# print(len(features))
# pickle_open = open("labels.pkl","rb")
# labels = pickle.load(pickle_open)
# pickle_open = open("feature.pkl","rb")
# features = pickle.load(pickle_open)

# features = features.reshape(-1,80,80,3)

# model = keras.Sequential([
#     keras.layers.Conv2D(filters=32, kernel_size=(3,3),padding="same", input_shape=(80,80,3),activation='relu'),
#     keras.layers.MaxPool2D((2,2)),
#     keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding="same", activation="relu"),
#     keras.layers.MaxPool2D((2,2)),
#     keras.layers.Conv2D(filters=128, kernel_size=(3,3),padding="same", activation="relu"),
#     keras.layers.MaxPool2D((2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(512,activation="relu"),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(2,activation="softmax")
# ])

# model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# model.fit(features, labels,epochs=15, batch_size=16)

# model.save("model.h5")

model = keras.models.load_model("model.h5")

# pickle_open = open("test.pkl","rb")
# test_data = pickle.load(pickle_open)
# # print(test_data)
# feature_test = []
# label_test = []
# count = 0
# for feature,label in test_data:
#     feature_test.append(feature)
#     label_test.append(label)
#     if count>1000:
#         break
#     count+=1

# feature_test = np.array(feature_test)
# label_test = np.array(label_test)
# feature_test = feature_test / 255
# feature_test = feature_test.reshape(-1,80,80,3)
# model.evaluate(feature_test, label_test)

img = cv2.imread("test8.jpg")
img = cv2.resize(img,(80,80))
plt.imshow(img)
plt.show()
img_array = np.array(img).flatten()

img_array = img_array/255
if (model.predict(img_array.reshape(1,80,80,3))[0][0]>model.predict(img_array.reshape(1,80,80,3))[0][1]):
    print("DOG")
else:
    print("cat")
# if (model.predict(img_array.reshape(1,128,128,3)))<0.5:
#     print("Dog")
# else:
#     print("cat")
