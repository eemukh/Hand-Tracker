import cv2
import mediapipe as mp
import time

# 1. Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 2. Boot up the Webcam
print("Starting webcam... Press 'q' to quit.")
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
pTime = 0

# 3. Setup the MediaPipe Hands Model
# model_complexity: 0 (Fastest, lowest accuracy) to 1 (Slowest, highest accuracy)
with mp_hands.Hands(
    model_complexity=0, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # --- Performance Optimization ---
        # Passing by reference improves performance. 
        # Mark the image as not writeable before passing it to the model.
        image.flags.writeable = False
        
        # MediaPipe requires RGB images, but OpenCV captures in BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 4. Run Inference
        results = hands.process(image)

        # --- Draw the Results ---
        # Mark the image as writeable again and convert back to BGR for OpenCV display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If hands were found in the frame
        if results.multi_hand_landmarks:
            # Loop through each hand detected
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Draw the skeleton on the image
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # --- Example: How to access a specific joint ---
                # Example: Index Finger Tip is Landmark #8
                # Coordinates are normalized (0.0 to 1.0), so we multiply by image dimensions
                h, w, c = image.shape
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                
                # Draw a prominent circle on the index fingertip
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # 5. Calculate and Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        
        # Add the FPS counter to the screen
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # 6. Show the video feed
        cv2.imshow('Hand Tracking', image)
        
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()