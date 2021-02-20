from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np

def classify_mnist():
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.0

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=30,
    validation_data = (X_valid, y_valid))





classify_mnist()
data = pd.read_csv("heart.csv")
X = data.iloc[:, 1:-1]
y =data.iloc[:,-1]
'''X = data.iloc[:, 2:-1]
print(X)
y = data.iloc[:, 1].replace('0=Blood Donor', 0).replace('3=Cirrhosis', 3).replace('0s=suspect Blood Donor',4).replace('1=Hepatitis', 1).replace('2=Fibrosis',2)
X.iloc[:, 1] = X.iloc[:, 1].astype('category')
X.iloc[:, 1] = X.iloc[:, 1].cat.codes
X = X.replace('3N9', 39)
X = X.fillna(0)'''
print(y)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,2), random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print("TRAIN:", accuracy_score(y_train_pred, y_train))
print("TEST:", accuracy_score(y_test_pred, y_test))
print(classification_report(y_train,y_train_pred))

print(classification_report(y_test,y_test_pred))



