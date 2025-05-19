import cv2
import numpy as np
import mediapipe as mp
import json
from pylsl import StreamInfo, StreamOutlet
import matplotlib.pyplot as plt
from collections import deque
import pickle as pkl

# Landmarks for left and right eyes (can be adjusted)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Define stream information: Name, Type, Number of channels, Rate, Format, Unique ID
info1 = StreamInfo('eyeLandmarks', 'Markers', 24, 0, 'float32', 'eyeLandmarks1234')
info2 = StreamInfo('EAR', 'Markers', 2, 0, 'float32', 'headpose1234')
info3 = StreamInfo('MediapipeFPS', 'Markers', 1, 0, 'string', 'FPS1234')
info4 = StreamInfo('faceLandmarks', 'Markers', 956, 0, 'float32', 'faceLandmarks1234')

outlet = StreamOutlet(info1)
outlet2 = StreamOutlet(info2)
outlet3 = StreamOutlet(info3)
outlet4 = StreamOutlet(info4)

def send_lsl_eyelandmarks(left_eye_landmarks, right_eye_landmarks):
    # Flatten the list of (x, y) tuples
    flat_left = [coord for point in left_eye_landmarks for coord in point]
    flat_right = [coord for point in right_eye_landmarks for coord in point]
    outlet.push_sample(flat_left + flat_right)

def send_lsl_EAR(left_ear, right_ear):
    outlet2.push_sample([float(left_ear), float(right_ear)])

def send_lsl_fps():
    outlet3.push_sample(["."])

def send_lsl_sendfacelandmarks(landmarks):
    flattened = [coord for tup in landmarks for coord in tup]  # [x1, y1, x2, y2, ...]
    outlet4.push_sample(flattened)  

# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye_landmarks):
    vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (vertical_1 + vertical_2) / (2.0 * horizontal)



def write_to_pkl(data, filename="datataran.pkl"):
    with open(filename, "wb") as file:
        pkl.dump(data, file)

# Initialize MediaPipe FaceMesh and VideoCapture
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
cap = cv2.VideoCapture(1)

#save all the landmarks in a list
landmarks_list = []
eyelandmark_list = []




eyes = {"left_ear": [], "right_ear": []}

# Main loop for capturing frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark]

            landmarks_list.append(landmarks)

            # Get eye landmarks
            left_eye_landmarks = [landmarks[i] for i in LEFT_EYE]
            right_eye_landmarks = [landmarks[i] for i in RIGHT_EYE]

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye_landmarks)
            right_ear = calculate_ear(right_eye_landmarks)


            eyes["left_ear"].append(left_ear)
            eyes["right_ear"].append(right_ear)

            eyelandmark_list.append([left_eye_landmarks, right_eye_landmarks])


            # Send EAR values to LSL
            try:
                send_lsl_eyelandmarks(left_eye_landmarks,right_eye_landmarks)
                send_lsl_fps()
                send_lsl_EAR(left_ear, right_ear)
                send_lsl_sendfacelandmarks(landmarks)
                
            except Exception as e:
                print(f"Error sending face landmarks: {e}")

    # Show the video feed
    cv2.imshow("Eye Blink Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

