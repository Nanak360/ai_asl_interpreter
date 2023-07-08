import csv
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import time
import numpy as np
import math

h1 = "CILOUV"
h2 = "ABDEFGHKMNPQRSTXYZWJ"
chars = [chr(i) for i in range(65, 91)]
imagesPerChar = 2500

f_1h = open("trainingData/singleHand.csv", "a+", newline='')
writer_f_1h = csv.writer(f_1h)
f_2h = open("trainingData/bothHand.csv", "a+", newline='')
writer_f_2h = csv.writer(f_2h)


def two_hands(handedness, _landmarks, character):
    hand_a = MessageToDict(handedness[0])["classification"][0]
    hand_b = MessageToDict(handedness[1])["classification"][0]
    temp_2 = [character]
    x_1, y_1, x_2, y_2 = [], [], [], []
    if hand_a["label"] != hand_b["label"]:
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
            temp_2.append(dist)
            temp_2.append(angle)
        writer_f_2h.writerow(temp_2)
        return True
    return False


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
while chars:

    print("Following characters are due -", chars)
    collect_for = input("Please enter the character you want to collect data for - ").upper()
    if collect_for not in chars:
        print("A to Z characters only!\n Start again")
        continue
    if collect_for in h1:
        mnh = 1
        text_string = collect_for + " - only RIGHT HAND"
    elif collect_for in h2:
        mnh = 2
        text_string = collect_for + " - BOTH HANDS"
    print("Hit esc to stop collecting data\nWill start capturing in...", end=" ")
    for t in range(2, -1, -1):
        print(t + 1, end=" ")
        if t == 1:
            hands = mp_hands.Hands(
                min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=mnh)
            continue
        time.sleep(.8)
    print("\nCapturing for", collect_for, end=" ")
    count_captured = 0
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
        count_text = str(count_captured) + " captured   " + str(
            imagesPerChar - count_captured) + " remaining"
        image = cv2.putText(image, text_string, (width // 16, height // 16 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2, cv2.LINE_4)
        image = cv2.putText(image, count_text, (width // 16, height // 16 + 60), cv2.FONT_HERSHEY_SIMPLEX, .8,
                            (50, 50, 250), 2, cv2.LINE_AA)
        image = cv2.putText(image, "'esc' key to EXIT", (width // 16, height // 3 + 45 + height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .7, (50, 250, 250), 2, cv2.LINE_4)
        if results.multi_hand_landmarks:
            if len(results.multi_handedness) == 2 and mnh == 2:
                if two_hands(results.multi_handedness, results.multi_hand_landmarks, collect_for):
                    count_captured += 1

            elif len(results.multi_handedness) == 1 and mnh == 1:
                hand = MessageToDict(results.multi_handedness[0])["classification"][0]
                landmarks = MessageToDict(results.multi_hand_landmarks[0])["landmark"]
                temp = [collect_for]

                for dic in landmarks:
                    temp.append(dic['x'])
                    temp.append(dic['y'])
                writer_f_1h.writerow(temp)
                count_captured += 1

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        ref_image = cv2.resize(cv2.imread("refImages/"+str(collect_for)+".jpg"), (width, height), interpolation=cv2.INTER_AREA)
        numpy_horizontal_concat = np.concatenate((ref_image, image), axis=1)
        cv2.imshow('MediaPipe Hands', numpy_horizontal_concat)
        if cv2.waitKey(5) & 0xFF == 27 or count_captured == imagesPerChar:
            break
    hands.close()
    cv2.destroyWindow("MediaPipe Hands")
    chars.remove(collect_for)
    print("- DONE\n\n")
f_2h.close()
f_1h.close()
