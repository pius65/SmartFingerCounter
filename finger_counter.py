import cv2
import mediapipe as mp

# Initialize Mediapipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to count the number of raised fingers
def count_fingers(hand_landmarks):
    fingers = []
    
    # Thumb
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers (index, middle, ring, pinky)
    for i in range(1, 5):
        if hand_landmarks.landmark[mp_hands.HandLandmark(i * 5)].y < hand_landmarks.landmark[mp_hands.HandLandmark(i * 5 - 2)].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

# Open the webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.9) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB and process it
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        # Convert the RGB image back to BGR
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw the hand annotations on the image and count fingers for each hand
        if results.multi_hand_landmarks:
            total_fingers = 10
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Count fingers for the current hand
                num_fingers = count_fingers(hand_landmarks)
                total_fingers += num_fingers

            # Display the total number of fingers raised
            cv2.putText(image, f'Total Fingers: {total_fingers}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Hand Tracking', image)

        # Check if 'q' key is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
