import cv2
import dlib
import math
import numpy as np
from tensorflow.keras.models import load_model

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(2, 3))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it  to the format (x, y, w, h) as we would normally do with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def get_landmarks(shape, rect):
    xlist = []
    ylist = []
    _, _, w, h = rect_to_bb(rect)
    for i in range(68):  # x and y coordinates
        x = 48 * float(float(shape.part(i).x - rect.left()) / w)
        y = 48 * float(float(shape.part(i).y - rect.top()) / h)
        xlist.append(x)
        ylist.append(y)
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
    return landmarks_vectorised


def emotion_detector(model, cam_id=0):
    cam = cv2.VideoCapture(cam_id)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        _, frame = cam.read()
        image = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rects = detector(image, 1)
        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(image, rect)
            coords = shape_to_np(shape)
            landmark_vect = np.expand_dims(np.array(get_landmarks(shape, rect)), axis=0)
            emotion_idx = np.argmax(model.predict(landmark_vect))
            # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the emotion
            cv2.putText(frame, emotions[emotion_idx], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
            for (x, y) in coords:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xff == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


fer_model = load_model('best_fer_model.h5')
emotion_detector(fer_model)
