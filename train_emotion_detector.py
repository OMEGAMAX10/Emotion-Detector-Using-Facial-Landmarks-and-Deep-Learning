import os
import cv2
import h5py
import dlib
import math
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(2, 3))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_landmarks(image):
    detections = detector(image, 1)
    landmark_list = []
    for detection in detections:
        shape = predictor(image, detection)  # Draw Facial Landmarks
        xlist = []
        ylist = []
        for i in range(68):  # x and y coordinates
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))
        landmark_list.append(landmarks_vectorised)
    return landmark_list


def make_dataset(train_filename, test_filename, dataset_folder="datasets", fer_dataset="fer2013.csv"):
    dataset_dir = Path(dataset_folder)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(fer_dataset)
    data_train = data[data.Usage == "Training"]
    data_test = data[data.Usage.str.contains("Test")]
    train_images = [get_landmarks(clahe.apply(np.reshape(pixels.split(" "), (48, 48)).astype("uint8"))) for pixels in data_train["pixels"]]
    test_images = [get_landmarks(clahe.apply(np.reshape(pixels.split(" "), (48, 48)).astype("uint8"))) for pixels in data_test["pixels"]]
    training_labels = data_train["emotion"].tolist()
    test_labels = data_test["emotion"].tolist()
    npar_train = np.array([landmark for landmark_list in train_images for landmark in landmark_list])
    npar_test = np.array([landmark for landmark_list in test_images for landmark in landmark_list])
    npar_train_labels = np.array([label for idx, label in enumerate(training_labels) if len(train_images[idx]) > 0])
    npar_test_labels = np.array([label for idx, label in enumerate(test_labels) if len(test_images[idx]) > 0])
    # Save dataset in .h5 files
    hf_train = h5py.File(os.path.join(dataset_dir, train_filename), 'w')
    hf_train.create_dataset('list_classes', data=emotions)
    hf_train.create_dataset('train_set_x', data=npar_train)
    hf_train.create_dataset('train_set_y', data=npar_train_labels)
    hf_test = h5py.File(os.path.join(dataset_dir, test_filename), 'w')
    hf_test.create_dataset('list_classes', data=emotions)
    hf_test.create_dataset('test_set_x', data=npar_test)
    hf_test.create_dataset('test_set_y', data=npar_test_labels)


def get_dataset(file_train, file_test):
    hf_train = h5py.File(file_train, 'r')
    npar_train = np.array(hf_train.get('train_set_x'))
    npar_train_labels = np.array(hf_train.get('train_set_y'))
    npar_train_labels = tf.one_hot(npar_train_labels, len(hf_train.get('list_classes'))).numpy()
    hf_test = h5py.File(file_test, 'r')
    npar_test = np.array(hf_test.get('test_set_x'))
    npar_test_labels = np.array(hf_test.get('test_set_y'))
    npar_test_labels = tf.one_hot(npar_test_labels, len(hf_test.get('list_classes'))).numpy()
    return npar_train, npar_train_labels, npar_test, npar_test_labels


def fer_model(input_shape):
    X_input = Input(input_shape)
    X = Dense(128, input_shape=input_shape, activation='relu', kernel_initializer='glorot_normal')(X_input)
    X = Dense(128, activation='relu', kernel_initializer='glorot_normal')(X)
    X = Dense(128, activation='relu', kernel_initializer='glorot_normal')(X)
    X = Dense(128, activation='relu', kernel_initializer='glorot_normal')(X)
    X = Dense(7, activation='softmax')(X)
    er_model = Model(inputs=X_input, outputs=X, name='fer_model')
    return er_model


def train_model(fer_train, fer_train_labels, fer_test, fer_test_labels, model_file='best_fer_model.h5', batch_size=64, epochs=300):
    model = fer_model(fer_train.shape[1:])
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    checkpoint = ModelCheckpoint(model_file, verbose=1, monitor='val_accuracy', save_best_only=True, mode='auto')
    model.fit(x=fer_train, y=fer_train_labels, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint],
              validation_data=(fer_test, fer_test_labels), shuffle=True)
    return model


# make_dataset("fer_train.h5", "fer_test.h5")
fer_train, fer_train_labels, fer_test, fer_test_labels = get_dataset("datasets/fer_train.h5", "datasets/fer_test.h5")
# model = train_model(fer_train, fer_train_labels, fer_test, fer_test_labels)
model = load_model("best_fer_model.h5")
model.summary()

preds = model.evaluate(fer_test, fer_test_labels)
print(f'Model Accuracy for Test Dataset: {preds[1] * 100} %\nModel Loss for Test Dataset: {preds[0]}')
