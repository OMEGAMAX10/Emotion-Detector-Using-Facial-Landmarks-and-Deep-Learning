import cv2
import glob
import h5py
import dlib
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # Emotion list
clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(2, 3))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_files(emotion):
    files_train = glob.glob("train/" + emotion + "/*")  # dataset
    training = files_train
    files_test = glob.glob("test/" + emotion + "/*")  # dataset
    prediction = files_test
    return training, prediction


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


def make_dataset(file_train, file_test):
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on " + emotion)
        training, prediction = get_files(emotion)
        for item in training:
            gray = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
            clahe_image = clahe.apply(gray)
            landmark_list = get_landmarks(clahe_image)
            for landmark in landmark_list:
                training_data.append(landmark)  # append image array to training data list
                training_labels.append(emotions.index(emotion))
        for item in prediction:
            gray = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
            clahe_image = clahe.apply(gray)
            landmark_list = get_landmarks(clahe_image)
            for landmark in landmark_list:
                prediction_data.append(landmark)
                prediction_labels.append(emotions.index(emotion))
    npar_train = np.array(training_data)
    npar_train_labels = np.array(training_labels)
    npar_pred = np.array(prediction_data)
    npar_pred_labels = np.array(prediction_labels)
    # Save dataset in .h5 files
    hf_train = h5py.File(file_train, 'w')
    hf_train.create_dataset('list_classes', data=emotions)
    hf_train.create_dataset('train_set_x', data=npar_train)
    hf_train.create_dataset('train_set_y', data=npar_train_labels)
    hf_test = h5py.File(file_test, 'w')
    hf_test.create_dataset('list_classes', data=emotions)
    hf_test.create_dataset('test_set_x', data=npar_pred)
    hf_test.create_dataset('test_set_y', data=npar_pred_labels)


def get_dataset(file_train, file_test):
    hf_train = h5py.File(file_train, 'r')
    npar_train = np.array(hf_train.get('train_set_x'))
    npar_train_labels = np.array(hf_train.get('train_set_y'))
    npar_train_labels = tf.one_hot(npar_train_labels, len(hf_train.get('list_classes'))).numpy()
    hf_test = h5py.File(file_test, 'r')
    npar_pred = np.array(hf_test.get('test_set_x'))
    npar_pred_labels = np.array(hf_test.get('test_set_y'))
    npar_pred_labels = tf.one_hot(npar_pred_labels, len(hf_test.get('list_classes'))).numpy()
    return npar_train, npar_train_labels, npar_pred, npar_pred_labels


def fer_model(input_shape):
    X_input = Input(input_shape)
    X = Dense(128, input_shape=input_shape, activation='relu', kernel_initializer='glorot_normal')(X_input)
    X = Dense(128, activation='relu', kernel_initializer='glorot_normal')(X)
    X = Dense(128, activation='relu', kernel_initializer='glorot_normal')(X)
    X = Dense(128, activation='relu', kernel_initializer='glorot_normal')(X)
    X = Dense(7, activation='softmax')(X)
    er_model = Model(inputs=X_input, outputs=X, name='fer_model')
    return er_model


def train_model(fer_train, fer_train_labels, fer_pred, fer_pred_labels, model_file='best_fer_model.h5', batch_size=64,
                epochs=1000):
    model = fer_model(fer_train.shape[1:])
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    checkpoint = ModelCheckpoint(model_file, verbose=1, monitor='val_accuracy', save_best_only=True, mode='auto')
    model.fit(x=fer_train, y=fer_train_labels, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint],
              validation_data=(fer_pred, fer_pred_labels), shuffle=True)
    return model


# make_dataset("datasets/fer_train.h5", "datasets/fer_test.h5")
fer_train, fer_train_labels, fer_pred, fer_pred_labels = get_dataset("datasets/fer_train.h5", "datasets/fer_test.h5")
# model = train_model(fer_train, fer_train_labels, fer_pred, fer_pred_labels)
model = load_model('best_fer_model.h5')
model.summary()

preds = model.evaluate(fer_pred, fer_pred_labels)
print(f'Model Accuracy for Test Dataset: {preds[1] * 100} %\nModel Loss for Test Dataset: {preds[0]}')
