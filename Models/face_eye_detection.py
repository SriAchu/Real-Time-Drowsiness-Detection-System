import cv2
import numpy as np
import time
import joblib
from skimage.feature import hog
import mediapipe as mp
import platform
import os
from pushbullet import Pushbullet

PB_ACCESS_TOKEN = "o.syCszdz5faj9cEQz730FOHJtRqDuBc5T"
pb = Pushbullet(PB_ACCESS_TOKEN)

EAR_THRESHOLD = 0.25
EYE_CLOSED_THRESH_SEC = 1.5
FORCE_EYE_CLOSED_THRESHOLD = 0.15
EYE_PROB_THRESHOLD = 0.6
EYE_CONSECUTIVE_FRAMES = 5
MIN_EYE_AREA = 500

EXPECTED_FEATURE_LENGTH = 1768
MAR_THRESHOLD = 0.50
MOUTH_YAWN_THRESH_SEC = 1.5
MOUTH_CONSECUTIVE_FRAMES = 3
MOUTH_PROB_THRESHOLD = 0.60
MIN_MOUTH_AREA = 500
FORCE_YAWN_THRESHOLD = 0.70

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 191, 402, 317, 82]

def play_alert_sound():
    try:
        if platform.system() == "Windows":
            os.system('PowerShell -Command "Add-Type –AssemblyName System.Speech; '
                     '$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                     '$speak.Speak(\'Drowsiness detected\');"')
        elif platform.system() == "Darwin":
            os.system('say "Drowsiness detected"')
        else:
            os.system('espeak "Drowsiness detected"')
    except Exception as e:
        print("Voice alert failed:", e)

def send_phone_alert():
    try:
        pb.push_note("Drowsiness Alert", "Drowsiness detected while driving!")
    except Exception as e:
        print("Failed to send phone alert:", e)

eye_model = joblib.load("eye_model_svm.pkl")
scaler_eye = joblib.load("scaler_eye.pkl")
mouth_model = joblib.load("mouth_model_svm.pkl")
scaler_mouth = joblib.load("scaler_mouth.pkl")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def compute_haar_features(img):
    integral = cv2.integral(img)
    h, w = img.shape
    step_size = 16
    regions = []
    for y in range(0, h - 32, step_size):
        for x in range(0, w - 32, step_size):
            region = integral[y:y+33, x:x+33]
            regions.append(region[-1,-1] - region[0,-1] - region[-1,0] + region[0,0])
    return np.array(regions)

def extract_features(img):
    gray = cv2.resize(img, (64, 64))
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    haar_feat = compute_haar_features(gray)
    combined = np.concatenate((hog_feat, haar_feat))
    return combined.reshape(1, -1) if len(combined) == EXPECTED_FEATURE_LENGTH else None

def eye_aspect_ratio(landmarks, eye_indices):
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    ear = (np.linalg.norm(points[1]-points[5]) + np.linalg.norm(points[2]-points[4])) / \
          (2 * np.linalg.norm(points[0]-points[3]))
    return ear

def mouth_aspect_ratio(landmarks):
    vert1 = np.array([landmarks[13].x, landmarks[13].y])
    vert2 = np.array([landmarks[14].x, landmarks[14].y])
    horiz1 = np.array([landmarks[78].x, landmarks[78].y])
    horiz2 = np.array([landmarks[308].x, landmarks[308].y])
    mar = (np.linalg.norm(vert1 - vert2) / np.linalg.norm(horiz1 - horiz2))
    return mar

def extract_crop(indices, frame, expansion=5):
    h, w = frame.shape[:2]
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords, y_coords = zip(*points)
    x1, x2 = max(min(x_coords)-expansion, 0), min(max(x_coords)+expansion, w)
    y1, y2 = max(min(y_coords)-expansion, 0), min(max(y_coords)+expansion, h)
    return frame[y1:y2, x1:x2]

cap = cv2.VideoCapture(0)
eye_pred_history = []
mouth_pred_history = []
eye_closed_start = None
mouth_open_start = None
alert_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    eye_state = "Open"
    mouth_state = "Closed"
    drowsy = False

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

                # 👀 EYE DETECTION (EAR + ML MODEL)
        # ====================
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2

        eye_crop = extract_crop(LEFT_EYE + RIGHT_EYE, frame)
        if eye_crop.size > 0 and eye_crop.shape[0] * eye_crop.shape[1] > MIN_EYE_AREA:
            eye_resized = cv2.resize(eye_crop, (128, 128))
            gray_eye = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2GRAY)
            gray_eye = cv2.equalizeHist(gray_eye)

            eye_features = extract_features(cv2.resize(gray_eye, (64, 64)))
            if eye_features is not None:
                eye_features = scaler_eye.transform(eye_features)
                eye_prob = eye_model.predict_proba(eye_features)[0][1]

                if eye_prob > EYE_PROB_THRESHOLD and ear < EAR_THRESHOLD:
                    eye_state = "Closed"
                    eye_closed_start = eye_closed_start or time.time()
                    if time.time() - eye_closed_start > EYE_CLOSED_THRESH_SEC:
                        drowsy = True
                else:
                    eye_state = "Open"
                    eye_closed_start = None
            else:
                eye_state = "Unknown"
                eye_closed_start = None
        else:
            eye_state = "Unknown"
            eye_closed_start = None

        # 🗣️ MOUTH DETECTION WITH MODEL
        mouth_crop = extract_crop(MOUTH[:4], frame)
        mar = mouth_aspect_ratio(landmarks)
        mouth_state = "Closed"

        if mouth_crop.size > 0 and mouth_crop.shape[0] * mouth_crop.shape[1] > MIN_MOUTH_AREA:
            mouth_resized = cv2.resize(mouth_crop, (128, 128))
            gray_mouth = cv2.cvtColor(mouth_resized, cv2.COLOR_BGR2GRAY)
            gray_mouth = cv2.equalizeHist(gray_mouth)

            mouth_features = extract_features(cv2.resize(gray_mouth, (64, 64)))
            if mouth_features is not None:
                mouth_features = scaler_mouth.transform(mouth_features)
                yawn_prob = mouth_model.predict_proba(mouth_features)[0][1]

                mouth_pred_history.append((mar, yawn_prob))
                if len(mouth_pred_history) > MOUTH_CONSECUTIVE_FRAMES:
                    mouth_pred_history.pop(0)

                avg_mar = np.mean([m[0] for m in mouth_pred_history])
                avg_prob = np.mean([m[1] for m in mouth_pred_history])

                if avg_mar > FORCE_YAWN_THRESHOLD:
                    mouth_state = "Yawning (Forced)"
                elif avg_mar > MAR_THRESHOLD and avg_prob > MOUTH_PROB_THRESHOLD:
                    mouth_state = "Yawning"
                else:
                    mouth_state = "Closed"

        if "Yawning" in mouth_state:
            mouth_open_start = mouth_open_start or time.time()
            if time.time() - mouth_open_start > MOUTH_YAWN_THRESH_SEC:
                drowsy = True
        else:
            mouth_open_start = None

    status = f"Eyes: {eye_state}  | Mouth: {mouth_state}"
    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
               (0, 255, 0) if not drowsy else (0, 0, 255), 2)

    if drowsy:
        cv2.putText(frame, "DROWSINESS ALERT!", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                   (0, 0, 255), 3)
        if not alert_triggered:
            play_alert_sound()
            send_phone_alert()
            alert_triggered = True
    else:
        alert_triggered = False

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
