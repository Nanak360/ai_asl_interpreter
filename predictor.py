import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import numpy as np
import math
import joblib
import os

# save_location = "captured_images"
classifier = joblib.load('svm_2h')
classifier_1h = joblib.load('svm_1h')
# os.mkdir(save_location)


def predict_res_1h(_landmarks):
    temp = []
    _landmarks = MessageToDict(_landmarks[0])["landmark"]
    for dic in _landmarks:
        temp.append(dic['x'])
        temp.append(dic['y'])
    return classifier_1h.predict([temp])[0]


def predict_res_2h(_landmarks):
    x_1, y_1, x_2, y_2, temp = [], [], [], [], []
    a_landmarks = MessageToDict(_landmarks[0])["landmark"]
    b_landmarks = MessageToDict(_landmarks[1])["landmark"]
    for d in a_landmarks:
        x_1.append(d['x'])
        y_1.append(d['y'])
    for d in b_landmarks:
        x_2.append(d['x'])
        y_2.append(d['y'])
    for i in range(21):
        dist = math.sqrt((x_2[i] - x_1[i]) ** 2 + (y_2[i] - y_1[i]) ** 2)
        if x_2[i] == x_1[i]:
            slope = float('inf')
        else:
            slope = (y_2[i] - y_1[i]) / (x_2[i] - x_1[i])
        angle = math.atan(slope)
        temp.append(dist)
        temp.append(angle)
    return classifier.predict([temp])[0]


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
image_count = 1
while cap.isOpened():
    success, image = cap.read()
    height, width = image.shape[:2]
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        if len(results.multi_handedness) == 2:
            char = predict_res_2h(results.multi_hand_landmarks)
            image = cv2.putText(image, char, (width // 16, height // 16 + 25), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255, 0, 0), 2, cv2.LINE_4)
        elif len(results.multi_handedness) == 1:
            char = predict_res_1h(results.multi_hand_landmarks)
            image = cv2.putText(image, char, (width // 16, height // 16 + 25), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255, 0, 0), 2, cv2.LINE_4)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # saving image part
        # img_name = os.path.join(save_location, "_" + char + "_" + str(image_count) + ".jpg")
        # image_count += 1
        # cv2.imwrite(img_name, image)

    ref_image = cv2.resize(cv2.imread("ref_sheet.jpg"), (width, height), interpolation=cv2.INTER_AREA)
    numpy_horizontal_concat = np.concatenate((ref_image, image), axis=1)
    cv2.imshow('MediaPipe Hands', numpy_horizontal_concat)
    # cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cv2.destroyWindow("MediaPipe Hands")
